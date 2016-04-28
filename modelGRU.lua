----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
-- Gated Recurrent Units (GRU) model
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'nn'
require 'nngraph'
require 'unsup'
require 'optim'
require 'torch'
require 'rnn'
local nninit = require 'nninit'

modelGRU = {};

----------------------------------------------------------------------
-- Handmade GRU
-- Creates one timestep of one GRU
-- Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
----------------------------------------------------------------------
function defineGRULayer(input_size, rnn_size, params)
  local n = params.n or 1
  local dropout = params.dropout or 0 
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  inputs[1] = nn.Identity()() -- x
  for L = 1,n do
    inputs[L+1] = nn.Identity()() -- prev_h[L]
  end
  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do

    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]; --OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    print(x);
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

local modelGRU, parent = torch.class('modelGRU', 'modelClass')

function modelGRU:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  -- Hidden layers
  for i = 1,structure.nLayers do
    -- Gated Reccurent Units
    if i == 1 then
      if (self.sequencer) then
        model:add(nn.GRU(self.windowSize, structure.layers[i], self.rho));
      else
        model:add(nn.GRU(structure.nInputs, structure.layers[i], self.rho));
      end
    else
      model:add(nn.GRU(structure.layers[i-1], structure.layers[i], self.rho));
    end
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
    gruModel = nn.Sequencer(model);
    model = nn.Sequential();
    -- Number of windows we will consider
    local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
    -- Here we add the subsequencing trick
    model:add(nn.SlidingWindow(2, self.windowSize, self.windowStep));
    model:add(gruModel);
    model:add(nn.JoinTable(2));
    model:add(nn.Linear(nWins * structure.nOutputs, structure.nOutputs));
  else
    -- Recursor case
    gruLayers = nn.Recursor(model);
    model = nn.Sequential();
    -- Add the GRU layers
    model:add(gruLayers);
    -- Needs to reshape the data from all outputs
    model:add(nn.Reshape(structure.layers[structure.nLayers]));
    -- And then add linear transform to number of classes
    model:add(nn.Linear(structure.layers[structure.nLayers], structure.nOutputs))
  end
  -- Return the complete model
  return model;
end

function modelGRU:definePretraining(structure, l, options)
  --[[ Encoder part ]]--
  local finalEncoder = nn.Sequential()
  local encoder = nn.Sequential();
  if l == 1 then 
    if (self.sequencer) then nIn = self.windowSize; else nIn = structure.nInputs end 
  else 
    nIn = structure.layers[l-1]; 
  end
  curGRU = nn.GRU(nIn, structure.layers[l], self.rho);
  -- Add the bias-adjusted LSTM to the network
  encoder:add(curGRU);
  -- Layer-wise linear transform
  if self.layerwiseLinear then encoder:add(nn.Linear(structure.layers[l], structure.layers[l])) end
  -- Batch normalization
  if self.batchNormalize then encoder:add(nn.BatchNormalization(structure.layers[l])); end
  -- Non-linearity
  if self.addNonLinearity then encoder:add(self.nonLinearity()); end
  -- Dropout
  if self.dropout then encoder:add(nn.Dropout(self.dropout)); end
  print(encoder);
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
  local decGRU = nn.GRU(structure.layers[l], structure.layers[l])
  decoder:add(nn.Sequencer(decGRU))
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

function modelGRU:retrieveEncodingLayer(model) 
  -- Here we only need to retrieve the encoding part
  return model.encoder;
end

function modelGRU:weightsInitialize(model)
  -- Find only the GRU modules
  linearNodes = model:findModules('nn.GRU')
  for l = 1,#linearNodes do
    modules = linearNodes[l]:findModules();
    for n = 1,#modules do
      module = modules[n];
      module:init('weight', self.initialize);
      module:init('bias', 1);
    end
  end
  -- Find only the linear modules
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

function modelGRU:weightsTransfer(model, trainedLayers)
  -- Find both LSTM and linear modules
  linearNodes = model:findModules('nn.Linear');
  -- Current linear layer
  local curLayer = 1;
  for l = 1,#trainedLayers do
    -- Find equivalent in pre-trained layer
    gruNodes = trainedLayers[l].encoder:findModules('nn.Linear');
    for k = 1,#gruNodes do
      linearNodes[curLayer].weights = gruNodes[k].weight;
      linearNodes[curLayer].bias = gruNodes[k].bias;
      curLayer = curLayer + 1;
    end
  end
  return model;
end

function modelGRU:parametersDefault()
  self.initialize = nninit.xavier;
  self.addNonLinearity = true;
  self.layerwiseLinear = true;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.sequencer = true;
  self.pretrain = true;
  self.dropout = 0.5;
  self.windowSize = 16;
  self.windowStep = 1;
  self.rho = 8;
end

function modelGRU:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end