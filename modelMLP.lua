----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Multi-layer perceptron
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'nn'
require 'torch'
require 'nninit'
require 'modelClass'

local modelMLP, parent = torch.class('modelMLP', 'modelClass')

function modelMLP:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  model:add(nn.Reshape(structure.nInputs));
  -- Hidden layers
  for i = 1,structure.nLayers do
    -- Linear transform
    if i == 1 then
      model:add(nn.Linear(structure.nInputs,structure.layers[i]));
    else
      model:add(nn.Linear(structure.layers[i-1],structure.layers[i]));
    end
    -- Batch normalization
    if self.batchNormalize then model:add(nn.BatchNormalization(structure.layers[i])); end
    -- Non-linearity
    model:add(self.nonLinearity());
    -- Dropout
    if self.dropout then model:add(nn.Dropout(self.dropout)); end
  end
  -- Final regression layer
  model:add(nn.Linear(structure.layers[structure.nLayers],structure.nOutputs)) 
  return model;
end

function modelMLP:definePretraining(inS, outS, options)
  --[[ Encoder part ]]--
  encoder = nn.Sequential();
  -- Linear transform
  encoder:add(nn.Linear(inS,outS));
  -- Batch normalization
  if self.batchNormalize then encoder:add(nn.BatchNormalization(outS)); end
  -- Non-linearity
  encoder:add(self.nonLinearity());
  -- Dropout
  if self.dropout then encoder:add(nn.Dropout(self.dropout)) end
  encoder:add(nn.Diag(outS));
  -- decoder
  decoder = nn.Sequential();
  decoder:add(nn.Linear(outS,inS));
  -- impose weight sharing
  decoder:get(1).weight = encoder:get(1).weight:t();
  decoder:get(1).gradWeight = encoder:get(1).gradWeight:t();
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, options.beta);
  return model;
end

function modelMLP:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  return model.encoder;
end

function modelMLP:weightsInitialize(model)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', self.initialize);
    module:init('bias', self.initialize);
  end
  return model;
end

function modelMLP:weightsTransfer(model, trainedLayers)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,trainedLayers do
    -- Find equivalent in pre-trained layer
    preTrained = trainedLayers[l]:findModules('nn.Linear');
    linearNodes[l].weight = preTrained[1].weight;
    linearNodes[l].bias = preTrained[1].bias;
  end
  return model;
end

function modelMLP:parametersDefault()
  self.initialize = nn.kaiming;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = false;
  self.pretrain = false;
  self.dropout = 0.5;
end

function modelMLP:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end