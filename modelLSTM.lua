----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'rnn'
require 'unsup'
require 'optim'
require 'torch'
require 'modelClass'
local nninit = require 'nninit'

----------------------------------------------------------------------
-- Sliding window module
--
-- This module takes a 1-dimensional time series as input and outputs all its sub-sequences of given size every given step
-- The optional tensOut (default false) decides if the output is either one large Tensor or a table of Tensors
-- The number of subsequences can be limited with parameter nf
----------------------------------------------------------------------

local SlidingWindow, parent = torch.class('nn.SlidingWindow','nn.Module')

function SlidingWindow:__init(tDim, size, step, nf, tensOut)
   parent.__init(self)
   self.tDim = tDim or 1
   self.size = size or 16
   self.step = step or 1
   self.nfeatures = nf or 1e9
   self.tensOut = tensOut or false
end

function SlidingWindow:updateOutput(input)
   local rep = torch.ceil((input:size(self.tDim) - self.size + 1) / self.step)
   local sz = torch.LongStorage(input:dim()+1)
   local currentOutput= {}
   if self.tensOut then currentOutput = torch.Tensor(rep, self.size) end
   for i=1,rep do
      currentOutput[i] = input:narrow(self.tDim, ((i - 1) * self.step + 1), self.size)
   end
   self.output = currentOutput
   return self.output
end

function SlidingWindow:updateGradInput(input, gradOutput)
   local slices = input:size(self.tDim)
   self.gradInput:resizeAs(input):zero()
   for i=1,#gradOutput do 
      local currentGradInput = gradOutput[i];
      local curIdx = ((i - 1) * self.step + 1);
      if (self.tDim == 1) then        
        self.gradInput[{{curIdx, curIdx + self.size - 1}}]:add(currentGradInput);
      else 
        self.gradInput[{{}, {curIdx, curIdx + self.size - 1}}]:add(currentGradInput);
      end
   end
   return self.gradInput
end

----------------------------------------------------------------------
-- Handmade LSTM
-- (From the Oxford course)
----------------------------------------------------------------------
function defineLSTMLayer(input_size, rnn_size, params)
  local n = params.n or 1
  local dropout = params.dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  -- Return final module
  return nn.gModule(inputs, outputs)
end

local modelLSTM, parent = torch.class('modelLSTM', 'modelClass')

function modelLSTM:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  -- Hidden layers
  for i = 1,structure.nLayers do
    -- Long Short-Term Memories
    if i == 1 then
      if (self.sequencer) then
        curLSTM = nn.FastLSTM(self.windowSize, structure.layers[i], self.rho);
      else
        curLSTM = nn.FastLSTM(structure.nInputs, structure.layers[i], self.rho);
      end
    else
      curLSTM = nn.FastLSTM(structure.layers[i-1], structure.layers[i], self.rho);
    end
    -- Always initialize the bias of the LSTM forget gate to 1 (trick from old RNN study)
    if self.initForget then curLSTM.i2g.bias[{{2*structure.layers[i]+1,3*structure.layers[i]}}]:fill(1) end
    -- Add the bias-adjusted LSTM to the network
    model:add(curLSTM);
    -- Layer-wise linear transform
    if self.layerwiseLinear then model:add(nn.Linear(structure.layers[i], structure.layers[i])) end
    -- Batch normalization
    if self.batchNormalize then model:add(nn.BatchNormalization(structure.layers[i])); end
    -- Non-linearity
    if self.addNonLinearity then model:add(self.nonLinearity()); end
    -- Dropout
    if self.dropout then model:add(nn.Dropout(self.dropout)); end
  end
  -- Final regression layer for classification
  if self.sequencer then 
    -- Sequencer case simply needs to add a linear transform to number of classes
    model:add(nn.Linear(structure.layers[structure.nLayers], structure.nOutputs))
    lstmModel = nn.Sequencer(model);
    model = nn.Sequential();
    -- Number of windows we will consider
    local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
    -- Here we add the subsequencing trick
    model:add(nn.SlidingWindow(2, self.windowSize, self.windowStep));
    model:add(lstmModel);
    model:add(nn.JoinTable(2));
    model:add(nn.Linear(nWins * structure.nOutputs, structure.nOutputs));
  else
    -- Recursor case
    lstmLayers = nn.Recursor(model);
    model = nn.Sequential();
    -- Add the LSTM layers
    model:add(lstmLayers);
    -- Needs to reshape the data from all outputs
    model:add(nn.Reshape(structure.layers[structure.nLayers]));
    -- And then add linear transform to number of classes
    model:add(nn.Linear(structure.layers[structure.nLayers], structure.nOutputs))
  end
  return model;
end

function modelLSTM:definePretraining(structure, l, options)
  --[[ Encoder part ]]--
  local finalEncoder = nn.Sequential()
  local encoder = nn.Sequential();
  if l == 1 then 
    if (self.sequencer) then nIn = self.windowSize; else nIn = structure.nInputs end 
  else 
    nIn = structure.layers[l-1]; 
  end
  curLSTM = nn.FastLSTM(nIn, structure.layers[l], self.rho);
  -- Always initialize the bias of the LSTM forget gate to 1 (trick from old RNN study)
  if self.initForget then curLSTM.i2g.bias[{{2*structure.layers[l]+1,3*structure.layers[l]}}]:fill(1) end
  -- Add the bias-adjusted LSTM to the network
  encoder:add(curLSTM);
  -- Layer-wise linear transform
  if self.layerwiseLinear then encoder:add(nn.Linear(structure.layers[l], structure.layers[l])) end
  -- Batch normalization
  if self.batchNormalize then encoder:add(nn.BatchNormalization(structure.layers[l])); end
  -- Non-linearity
  if self.addNonLinearity then encoder:add(self.nonLinearity()); end
  -- Dropout
  if self.dropout then encoder:add(nn.Dropout(self.dropout)); end
  -- Perform recurrent encoder
  encoder = nn.Sequencer(encoder);
  -- In the first layer we have to perform a Sliding Window
  if l == 1 then
    -- Number of windows we will consider
    local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
    -- Here we add the subsequencing trick
    finalEncoder:add(nn.SlidingWindow(2, self.windowSize, self.windowStep));
  end
  finalEncoder:add(encoder);  
  --[[ Decoder part ]]--
  local decoder = nn.Sequential()
  local finalDecoder = nn.Sequential()
  local decLSTM = nn.FastLSTM(structure.layers[l], structure.layers[l])
  decoder:add(nn.Sequencer(decLSTM))
  decoder:add(nn.Sequencer(nn.Linear(structure.layers[l], nIn)))
  finalDecoder:add(decoder);
  -- In the first layer we have to join windows
  if l == 1 then
    -- Number of windows we will consider
    local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
    -- Here we add the subsequencing trick
    finalDecoder:add(nn.JoinTable(2));
    finalDecoder:add(nn.Linear(nWins * self.windowSize, structure.nInputs));
  end
  -- Construct an autoencoder
  local model = unsup.AutoEncoder(finalEncoder, finalDecoder, options.beta);
  -- We will need a sequencer criterion for deeper layers
  if (l > 1) then model.loss = nn.SequencerCriterion(model.loss); end
  -- Return the complete model
  return model;
end

function modelLSTM:retrieveEncodingLayer(model)
  -- Retrieve only the encoding layer 
  encoder = model.encoder
  return encoder
end

function modelLSTM:weightsInitialize(model)
  -- Find only the linear modules (including LSTM's)
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', self.initialize);
    module:init('bias', self.initialize);
  end
  -- Initialize the batch normalization layers
  for k,v in pairs(model:findModules('nn.BatchNormalization')) do
    v.weight:fill(1)
    v.bias:zero()
  end
  return model;
end

function modelLSTM:weightsTransfer(model, trainedLayers)
  -- Find both LSTM and linear modules
  linearNodes = model:findModules('nn.Linear');
  -- Current linear layer
  local curLayer = 1;
  for l = 1,#trainedLayers do
    -- Find equivalent in pre-trained layer
    lstmNodes = trainedLayers[l].encoder:findModules('nn.Linear');
    for k = 1,#lstmNodes do
      linearNodes[curLayer].weights = lstmNodes[k].weight;
      linearNodes[curLayer].bias = lstmNodes[k].bias;
      curLayer = curLayer + 1;
    end
  end
  return model;
end

function modelLSTM:parametersDefault()
  self.initialize = nninit.kaiming;
  self.addNonLinearity = true;
  self.layerwiseLinear = true;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.initForget = true;
  self.sequencer = true;
  self.pretrain = true;
  self.dropout = 0.5;
  self.windowSize = 16;
  self.windowStep = 1;
  self.rho = 5;
end

function modelLSTM:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end