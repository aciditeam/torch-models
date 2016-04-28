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
require 'modelClass'
require 'modelCNN'
local nninit = require 'nninit'

local modelNIN, parent = torch.class('modelNIN', 'modelCNN')

----------------------------------------------------------------------
-- Network in Network
----------------------------------------------------------------------
function modelNIN:defineModel(structure, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  local model = nn.Sequential() 
  -- Reshape the input to fit the convolution
  model:add(nn.Reshape(structure.nInputs, 1));
  model:add(nn.Padding(2, -1));
  model:add(nn.Padding(2, 1)); 
  -- Convolution Layers
  model:add(nn.TemporalConvolution(1, 128, 5, 3))
  -- TODO
  -- Number of windows we will consider
  -- TODO
  local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
  if self.batchNormalize then
    model:add(nn.Reshape(math.floor(structure.nInputs / 3) * 128)); 
    model:add(nn.BatchNormalization(math.floor(structure.nInputs / 3) * 128));
    model:add(nn.Reshape(math.floor(structure.nInputs / 3), 128)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(128, 96, 1, 1))
  if self.batchNormalize then
    model:add(nn.Reshape(math.floor(structure.nInputs / 3) * 96)); 
    model:add(nn.BatchNormalization(math.floor(structure.nInputs / 3) * 96));
    model:add(nn.Reshape(math.floor(structure.nInputs / 3), 96)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.TemporalMaxPooling(3, 2))
  model:add(nn.Dropout(0.25))
  model:add(nn.TemporalConvolution(96, 192, 5, 2))
  if self.batchNormalize then
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5)) * 192)); 
    model:add(nn.BatchNormalization(math.floor(structure.nInputs / (3 * 5)) * 192));
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5)), 192)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(192, 256, 1, 1))
  if self.batchNormalize then
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5)) * 256)); 
    model:add(nn.BatchNormalization(math.floor(structure.nInputs / (3 * 5)) * 256));
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5)), 256)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.TemporalMaxPooling(3, 2))
  model:add(nn.Dropout(0.5))
  model:add(nn.TemporalConvolution(256, 256, 3))
  if self.batchNormalize then
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5 * 2 * 3)) * 256)); 
    model:add(nn.BatchNormalization(math.floor(structure.nInputs / (3 * 5 * 2 * 3)) * 256));
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5 * 2 * 3)), 256)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(256, 1024, 1, 1))
  if self.batchNormalize then
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5 * 2 * 3)) * 1024)); 
    model:add(nn.BatchNormalization(math.floor(structure.nInputs / (3 * 5 * 2 * 3)) * 1024));
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5 * 2 * 3)), 1024)); 
  end
  model:add(self.nonLinearity())
  -- Global Average Pooling Layer
  local final_mlpconv_layer = nn.TemporalConvolution(1024, 100, 1, 1)
  model:add(final_mlpconv_layer)
  if self.batchNormalize then
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5 * 2 * 3)) * 100)); 
    model:add(nn.BatchNormalization(math.floor(structure.nInputs / (3 * 5 * 2 * 3)) * 100));
    model:add(nn.Reshape(math.floor(structure.nInputs / (3 * 5 * 2 * 3)), 100)); 
  end
  model:add(self.nonLinearity())
  --model:add(nn.MyTemporalAveragePooling(10, 5, 5))
  model:add(nn.Reshape(100))
  model:add(nn.Linear(100, structure.nOutputs));
  -- all initial values in final layer must be a positive number.
  -- this trick is awfully important ('-')b
  final_mlpconv_layer.weight:abs()
  final_mlpconv_layer.bias:abs()
  return model
end

function modelNIN:defineStructure()
  -- Defined the fixed structure
  structure = {};
  -- Convolution Layers
  structure.nLayers = 7;
  structure.kernelSizes = {128, 96, 192, 256, 256, 1024, 100};
  structure.kernelWidth = {5, 1, 5, 1, 3, 1, 1};
  structure.kernelStep = {3, 1, 2, 1, 1, 1, 1};
  structure.divideFactor = {3, 5, 2, 3, 1, 1};
  structure.poolWidth = {1, 3, 1, 3, 1, 1};
  structure.poolStep = {1, 2, 1, 2, 1, 1};
  return structure; 
end

function modelNIN:definePretraining(structure, l, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- Get the fixed structure
  local fixedStructure = self:defineStructure();
  --[[ Encoder part ]]--
  encoder = nn.Sequential()
  -- Prepare the layer properties
  if (l == 1) then
    inS = 1; inSize = structure.nInputs;
    encoder:add(nn.Reshape(structure.nInputs, 1));
    -- Eventual padding
    encoder:add(nn.Padding(2, -1)); encoder:add(nn.Padding(2, 1));
    outS = fixedStructure.kernelSizes[l];
  else
    inS = kernelSize[l - 1];
    inSize = structure.nInputs;
    for i = 1,l do inSize = inSize / fixedStructure.divideFactor[l]; end
  end
  -- TODO
  -- TODO
  -- Number of windows we will consider
  local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
  -- TODO
  -- TODO
  -- Perform convolution
  encoder:add(nn.TemporalConvolution(inS, outS, fixedStructure.kernelWidth[l], fixedStructure.kernelStep[l]));
  -- Batch normalization
  if self.batchNormalize then
    curTPoints = structure.nInputs
    for i = 2,l do curTPoints = curTPoints / fixedStructure.divideFactor[l]; end
    encoder:add(nn.Reshape(curTPoints * outS)); 
    encoder:add(nn.BatchNormalization(curTPoints * outS));
    encoder:add(nn.Reshape(curTPoints, outS))
  end
  -- Non-linearity
  encoder:add(self.nonLinearity());
  -- Pooling
  encoder:add(nn.TemporalMaxPooling(fixedStructure.poolWidth[1], fixedStructure.poolStep[1]));
  -- Decoder:
  decoder = nn.Sequential()
  -- Put de-convolution
  curTPoints = structure.nInputs
  for i = 1,l do curTPoints = curTPoints / fixedStructure.divideFactor[l]; end
  decoder:add(nn.Reshape(curTPoints * fixedStructure.kernelSizes[l]))
  decoder:add(nn.Linear(curTPoints * fixedStructure.kernelSizes[l], inSize * inS))
  decoder:add(nn.Reshape(inSize, inS))
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, options.beta)
  return model;
end

function modelNIN:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder;
  --encoder:remove();
  return encoder;
end

function modelNIN:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.RReLU;
  self.batchNormalize = true;
  self.pretrain = true;
  self.dropout = 0.5;
end

function modelNIN:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end
