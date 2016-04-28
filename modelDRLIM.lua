----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'nn'
require 'nngraph'
require 'unsup'
require 'optim'
require 'torch'
local nninit = require 'nninit'

local modelDRLIM, parent = torch.class('modelDRLIM', 'modelSiamese')

----------------------------------------------------------------------
-- Handmade DRLIM
----------------------------------------------------------------------
function modelDRLIM:defineModel(structure, options)
  
  --Encoder 
  local encoder = nn.Sequential()
  encoder:add(nn.Linear(im_size,network_dim[1]))
  encoder:get(encoder:size()).bias:add(-encoder:get(encoder:size()).bias:min())
  encoder:add(nn.Threshold())
  encoder:add(nn.Linear(network_dim[1],network_dim[2])) 
  encoder:get(encoder:size()).bias:add(-encoder:get(encoder:size()).bias:min())
  encoder:add(nn.Threshold())
  encoder:add(nn.Linear(network_dim[2],network_dim[3])) 
  encoder:cuda()

  -- Full network --> Split two input images
  local model = nn.Sequential()

  -- Create the parallel siamese network and add it to the full network
  local encoder_siamese = nn.ParallelTable()
  encoder_siamese:add(encoder)
  encoder_siamese:add(encoder:clone('weight','bias','gradWeight','gradBias'))
  model:add(encoder_siamese) 

  -- Create the L2 distance function and add it to the full network
  local dist = nn.PairwiseDistance(2)
  model:add(dist)
  -- Criterion
  local criterion = nn.HingeEmbeddingCriterion(margin):cuda() 

  --Necessary (Torch bug) 
  dist.gradInput[1] = dist.gradInput[1]:cuda()
  dist.gradInput[2] = dist.gradInput[2]:cuda()

  return model, criterion, encoder;
end

function modelDRLIM:definePretraining(structure, l, options)
  -- TODO
  return model;
end

function modelDRLIM:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder
  encoder:remove();
  return model.encoder;
end

function modelDRLIM:weightsInitialize(model)
  -- TODO
  return model;
end

function modelDRLIM:weightsTransfer(model, trainedLayers)
  -- TODO
  return model;
end

function modelDRLIM:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.pretrainType = 'ae';
  self.pretrain = true;
  self.dropout = 0.5;
end

function modelDRLIM:parametersRandom()
  -- All possible non-linearities
  self.distributions = {};
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
  self.distributions.batchNormalize = {true, false};
  self.distributions.pretrainType = {'ae', 'psd'};
  self.distributions.pretrain = {true, false};
  self.distributions.dropout = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
end