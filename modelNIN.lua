----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'unsup'
require 'optim'
require 'torch'
require 'nninit'
require 'modelClass'
require 'modelCNN'

local modelNIN, parent = torch.class('modelNIN', 'modelCNN')

----------------------------------------------------------------------
-- Network in Network
----------------------------------------------------------------------
function modelNIN:defineModel(structure, options) --.defineNIN(input_size, rnn_size, params)
  local model = nn.Sequential() 
  -- Reshape the input to fit the convolution
  model:add(nn.Reshape(structure.nInputs, 1));
  model:add(nn.Padding(1, -1));
  model:add(nn.Padding(1, 1)); 
  -- Convolution Layers
  model:add(nn.TemporalConvolution(1, 128, 5, 3))
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(128, 96, 1, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalMaxPooling(3, 2))
  model:add(nn.Dropout(0.25))
  model:add(nn.TemporalConvolution(96, 192, 5, 2))
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(192, 256, 1, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalMaxPooling(3, 2))
  model:add(nn.Dropout(0.5))
  model:add(nn.TemporalConvolution(256, 256, 3))
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(256, 1024, 1, 1))
  model:add(self.nonLinearity())
  -- Global Average Pooling Layer
  local final_mlpconv_layer = nn.TemporalConvolution(1024, 10, 1, 1)
  model:add(final_mlpconv_layer)
  model:add(self.nonLinearity())
  --model:add(nn.MyTemporalAveragePooling(10, 5, 5))
  model:add(nn.Reshape(7 * 10))
  model:add(nn.Linear(7 * 10, structure.nOutputs));
  -- all initial values in final layer must be a positive number.
  -- this trick is awfully important ('-')b
  final_mlpconv_layer.weight:abs()
  final_mlpconv_layer.bias:abs()
  return model
end

function modelNIN:definePretraining(inS, outS, options)
  --
end

function modelNIN:retrieveEncodingLayer(model) 
  --
end

function modelNIN:parametersDefault()
  self.initialize = nn.kaiming;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = false;
  self.pretrain = false;
  self.dropout = 0.5;
end

function modelNIN:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end
