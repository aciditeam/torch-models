----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'modelCNN'
require 'unsup'
require 'optim'
local nninit = require 'nninit'

local modelVGG, parent = torch.class('modelVGG', 'modelCNN')

----------------------------------------------------------------------
-- Temporal version of VGG
-- 
----------------------------------------------------------------------
function modelVGG:defineModel(structure, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  local model = nn.Sequential() 
  local final_mlpconv_layer = nil
  -- Convolution Layers
  kW = 16; dW = 1;
  model:add(nn.Reshape(structure.nInputs, 1));
  if self.padding then model:add(nn.Padding(2, -(kW/2 - 1))); model:add(nn.Padding(2, kW/2)); end
  model:add(nn.TemporalConvolution(1, 64, kW, dW));
  if self.batchNormalize then
    model:add(nn.Reshape(structure.nInputs * 64)); 
    model:add(nn.BatchNormalization(structure.nInputs * 64));
    model:add(nn.Reshape(structure.nInputs, 64)); 
  end
  model:add(self.nonLinearity());
  if self.padding then model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1)); end
  model:add(nn.TemporalConvolution(64, 64, 3, 1));
  if self.batchNormalize then
    model:add(nn.Reshape(structure.nInputs * 64)); 
    model:add(nn.BatchNormalization(structure.nInputs * 64));
    model:add(nn.Reshape(structure.nInputs, 64)); 
  end
  model:add(self.nonLinearity());
  model:add(nn.TemporalMaxPooling(2, 2));
  model:add(nn.Dropout(0.25));
  -- Second layers
  if self.padding then model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1)); end
  model:add(nn.TemporalConvolution(64, 128, 3, 1))
  if self.batchNormalize then
    model:add(nn.Reshape((structure.nInputs / 2) * 128)); 
    model:add(nn.BatchNormalization((structure.nInputs / 2) * 128));
    model:add(nn.Reshape((structure.nInputs / 2), 128)); 
  end
  model:add(self.nonLinearity())
  if self.padding then model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1)); end
  model:add(nn.TemporalConvolution(128, 128, 3, 1))
  if self.batchNormalize then
    model:add(nn.Reshape((structure.nInputs / 2) * 128)); 
    model:add(nn.BatchNormalization((structure.nInputs / 2) * 128));
    model:add(nn.Reshape((structure.nInputs / 2), 128)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.TemporalMaxPooling(2, 2));
  model:add(nn.Dropout(0.25))
  -- Third layers 
  if self.padding then model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1)); end
  model:add(nn.TemporalConvolution(128, 256, 3, 1))
  if self.batchNormalize then
    model:add(nn.Reshape((structure.nInputs / 4) * 256)); 
    model:add(nn.BatchNormalization((structure.nInputs / 4) * 256));
    model:add(nn.Reshape((structure.nInputs / 4), 256)); 
  end
  model:add(self.nonLinearity())
  if self.padding then model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1)); end
  model:add(nn.TemporalConvolution(256, 256, 3, 1))
  if self.batchNormalize then
    model:add(nn.Reshape((structure.nInputs / 4) * 256)); 
    model:add(nn.BatchNormalization((structure.nInputs / 4) * 256));
    model:add(nn.Reshape((structure.nInputs / 4), 256)); 
  end
  model:add(self.nonLinearity())
  if self.padding then model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1)); end
  model:add(nn.TemporalConvolution(256, 256, 3, 1))
  if self.batchNormalize then
    model:add(nn.Reshape((structure.nInputs / 4) * 256)); 
    model:add(nn.BatchNormalization((structure.nInputs / 4) * 256));
    model:add(nn.Reshape((structure.nInputs / 4), 256)); 
  end
  model:add(self.nonLinearity())
  if self.padding then model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1)); end
  model:add(nn.TemporalConvolution(256, 256, 3, 1))
  if self.batchNormalize then
    model:add(nn.Reshape((structure.nInputs / 4) * 256)); 
    model:add(nn.BatchNormalization((structure.nInputs / 4) * 256));
    model:add(nn.Reshape((structure.nInputs / 4), 256)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.TemporalMaxPooling(2, 2));
  model:add(nn.Dropout(0.25))
  -- Fully connected Layers
  if self.padding then model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1)); end
  model:add(nn.TemporalConvolution(256, 1024, 3, 1))
  if self.batchNormalize then
    model:add(nn.Reshape((structure.nInputs / 8) * 1024)); 
    model:add(nn.BatchNormalization((structure.nInputs / 8) * 1024));
    model:add(nn.Reshape((structure.nInputs / 8), 1024)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.Dropout(0.5))
  model:add(nn.TemporalConvolution(1024, 1024, 1, 1))
  if self.batchNormalize then
    model:add(nn.Reshape((structure.nInputs / 8) * 1024)); 
    model:add(nn.BatchNormalization((structure.nInputs / 8) * 1024));
    model:add(nn.Reshape((structure.nInputs / 8), 1024)); 
  end
  model:add(self.nonLinearity())
  model:add(nn.Dropout(0.5))
  -- Final layer
  model:add(nn.TemporalConvolution(1024, structure.nOutputs, 1, 1))
  model:add(nn.Reshape((structure.nInputs / 8) * structure.nOutputs))
  model:add(nn.Linear((structure.nInputs / 8) * structure.nOutputs, structure.nOutputs))
  return model
end

function modelVGG:definePretraining(structure, l, options)
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
  else
    inS = fixedStructure.kernelSizes[l - 1];
    inSize = structure.nInputs;
    for i = 1,(l-1) do inSize = inSize / fixedStructure.poolWidth[i]; end
  end
  outS = fixedStructure.kernelSizes[l];
  -- In VGG we always pad
  encoder:add(nn.Padding(2, -(math.ceil(fixedStructure.kernelWidth[l] / 2) - 1))); encoder:add(nn.Padding(2, (fixedStructure.kernelWidth[l] / 2)));
  -- Perform convolution
  encoder:add(nn.TemporalConvolution(inS, outS, fixedStructure.kernelWidth[l], fixedStructure.kernelStep[l]));
  -- Batch normalization
  if self.batchNormalize then
    curTPoints = structure.nInputs
    for i = 1,(l-1) do curTPoints = curTPoints / fixedStructure.poolWidth[i]; end
    encoder:add(nn.Reshape(curTPoints * outS)); 
    encoder:add(nn.BatchNormalization(curTPoints * outS));
    encoder:add(nn.Reshape(curTPoints, outS))
  end
  -- Non-linearity
  encoder:add(self.nonLinearity());
  -- Pooling
  encoder:add(nn.TemporalMaxPooling(fixedStructure.poolWidth[l], fixedStructure.poolStep[l]));
  -- Decoder:
  decoder = nn.Sequential()
  -- Put de-convolution
  curTPoints = structure.nInputs
  for i = 1,l do curTPoints = curTPoints / fixedStructure.poolWidth[i]; end
  decoder:add(nn.Reshape(curTPoints * fixedStructure.kernelSizes[l]))
  decoder:add(nn.Linear(curTPoints * fixedStructure.kernelSizes[l], inSize * inS))
  decoder:add(nn.Reshape(inSize, inS))
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, options.beta)
  return model;
end

function modelVGG:defineStructure()
  -- Defined the fixed structure
  local structure = {};
  -- Convolution Layers
  structure.nLayers = 9;
  structure.kernelSizes = {64, 64, 128, 128, 256, 256, 256, 256, 1024};
  structure.kernelWidth = {16, 3, 3, 3, 3, 3, 3, 3, 3, 1};
  structure.kernelStep = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  structure.poolWidth = {1, 2, 1, 2, 1, 1, 1, 2, 1};
  structure.poolStep = {1, 2, 1, 2, 1, 1, 1, 2, 1};
  return structure; 
end

function modelVGG:defineSpatial(input_size, rnn_size, params)
  local model = nn.Sequential() 
   local final_mlpconv_layer = nil
   -- Convolution Layers
   model:add(ccn2.SpatialConvolution(3, 64, 3, 1, 1))
   model:add(self.nonLinearity())
   model:add(ccn2.SpatialConvolution(64, 64, 3, 1, 1))
   model:add(self.nonLinearity())
   model:add(ccn2.SpatialMaxPooling(2, 2))
   model:add(nn.Dropout(0.25))
      
   model:add(ccn2.SpatialConvolution(64, 128, 3, 1, 1))
   model:add(self.nonLinearity())
   model:add(ccn2.SpatialConvolution(128, 128, 3, 1, 1))
   model:add(self.nonLinearity())
   model:add(ccn2.SpatialMaxPooling(2, 2))
   model:add(nn.Dropout(0.25))
   
   model:add(ccn2.SpatialConvolution(128, 256, 3, 1, 1))
   model:add(self.nonLinearity())
   model:add(ccn2.SpatialConvolution(256, 256, 3, 1, 1))
   model:add(self.nonLinearity())
   model:add(ccn2.SpatialConvolution(256, 256, 3, 1, 1))
   model:add(self.nonLinearity())
   model:add(ccn2.SpatialConvolution(256, 256, 3, 1, 1))
   model:add(self.nonLinearity())
   model:add(ccn2.SpatialMaxPooling(2, 2))
   model:add(nn.Dropout(0.25))
   
   -- Fully Connected Layers   
   model:add(ccn2.SpatialConvolution(256, 1024, 3, 1, 0))
   model:add(self.nonLinearity())
   model:add(nn.Dropout(0.5))
   model:add(ccn2.SpatialConvolution(1024, 1024, 1, 1, 0))
   model:add(self.nonLinearity())
   model:add(nn.Dropout(0.5))
   
   model:add(nn.Transpose({4,1},{4,2},{4,3}))
   
   model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1))
   model:add(nn.Reshape(10))
   model:add(nn.SoftMax())

   return model
end

function modelVGG:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.RReLU;
  self.batchNormalize = true;
  self.kernelWidth = {};
  self.pretrain = true;
  self.padding = true;
  self.dropout = 0.5;
end

function modelVGG:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end