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
local nninit = require 'nninit'

modelRNN = {};

----------------------------------------------------------------------
-- Handmade RNN
-- (From the Oxford course)
----------------------------------------------------------------------
function defineRNNLayer(input_size, rnn_size, params)
  local n = params.n or 1
  local dropout = params.dropout or 0
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    local prev_h = inputs[L+1]
    if L == 1 then 
      --x = OneHot(input_size)(inputs[1])
      x = inputs[1];
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})
    table.insert(outputs, next_h)
  end
  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  -- Creates the final module
  return nn.gModule(inputs, outputs)
end

local modelRNN, parent = torch.class('modelRNN', 'modelClass')

function modelRNN:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  -- Hidden layers
  for i = 1,structure.nLayers do
    if i == 1 then nIn = self.windowSize; else nIn = structure.layers[i - 1]; end
    -- Prepare one layer of reccurent computation
    local r = nn.Recurrent(
      structure.layers[i], 
      nn.Linear(nIn, structure.layers[i]), 
      nn.Linear(structure.layers[i], structure.layers[i]), 
      self.nonLinearity(),
      self.rho
    );
    model:add(r);
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
    rnnModel = nn.Sequencer(model);
    model = nn.Sequential();
    -- Number of windows we will consider
    local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
    -- Here we add the subsequencing trick
    model:add(nn.SlidingWindow(2, self.windowSize, self.windowStep));
    model:add(rnnModel);
    model:add(nn.JoinTable(2));
    model:add(nn.Linear(nWins * structure.nOutputs, structure.nOutputs));
  else
    -- Recursor case
    rnnLayers = nn.Recursor(model, self.rho);
    model = nn.Sequential()
    -- Add the recurrent 
    model:add(rnnLayers);
    -- Needs to reshape the data from all outputs
    model:add(nn.Reshape(structure.layers[structure.nLayers]));
    -- And then add linear transform to number of classes
    model:add(nn.Linear(structure.layers[structure.nLayers], structure.nOutputs))
  end
  return model;
end

function modelRNN:definePretraining(structure, l, options)
  --[[ Encoder part ]]--
  local finalEncoder = nn.Sequential()
  local encoder = nn.Sequential();
  if l == 1 then 
    if (self.sequencer) then nIn = self.windowSize; else nIn = structure.nInputs end 
  else 
    nIn = structure.layers[l-1]; 
  end
   -- Prepare one layer of reccurent computation
  local r = nn.Recurrent(
    structure.layers[l], 
    nn.Linear(nIn, structure.layers[l]), 
    nn.Identity(), 
    self.nonLinearity(),
    1e9--self.rho
  );
  -- Add the RNN modules to the network
  encoder:add(r);
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
  -- Prepare decoding layer of reccurent computation
  local rDec = nn.Recurrent(
    structure.layers[l], 
    nn.Linear(structure.layers[l],structure.layers[l]), 
    nn.Identity(), 
    self.nonLinearity(),
    1e9--self.rho
  );
  decoder:add(nn.Sequencer(rDec));
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

function modelRNN:retrieveEncodingLayer(model)
  -- Retrieve only the encoding layer 
  encoder = model.encoder
  return encoder
end

function modelRNN:weightsInitialize(model)
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

function modelRNN:weightsTransfer(model, trainedLayers)
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

function modelRNN:parametersDefault()
  self.initialize = nninit.xavier;
  self.addNonLinearity = true;
  self.layerwiseLinear = true;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.sequencer = true;
  self.pretrain = true;
  self.windowSize = 16;
  self.windowStep = 1;
  self.dropout = 0.5;
  self.rho = 4;
end

function modelRNN:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end