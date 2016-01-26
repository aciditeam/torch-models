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
--require 'torch'
--require 'cunn'
--require 'ccn2'

-- TODO
-- If cuda, then need to switch between
-- * nn vs cunn
-- * cudnn ... (no temporal)
-- * ccn2 ?
-- TODO

local modelVGG, parent = torch.class('modelVGG', 'modelCNN')

----------------------------------------------------------------------
-- Temporal version of VGG
-- 
----------------------------------------------------------------------
function modelVGG:defineModel(structure, options)
  local model = nn.Sequential() 
  local final_mlpconv_layer = nil
  -- Convolution Layers
  kW = 16; dW = 1;
  model:add(nn.Reshape(structure.nInputs, 1));
  if self.padding then model:add(nn.Padding(1, -(kW/2 - 1))); model:add(nn.Padding(1, kW/2)); end
  model:add(nn.TemporalConvolution(1, 64, kW, dW));
  model:add(self.nonLinearity());
  model:add(nn.TemporalConvolution(64, 64, 3, 1));
  model:add(self.nonLinearity());
  model:add(nn.TemporalMaxPooling(2, 2));
  model:add(nn.Dropout(0.25));
  -- Second layers
  model:add(nn.TemporalConvolution(64, 128, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(128, 128, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalMaxPooling(2, 2));
  model:add(nn.Dropout(0.25))
  -- Third layers 
  model:add(nn.TemporalConvolution(128, 256, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(256, 256, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(256, 256, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(256, 256, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalMaxPooling(2, 2));
  model:add(nn.Dropout(0.25))
  -- Fully connected Layers
  model:add(nn.TemporalConvolution(256, 1024, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.Dropout(0.5))
  model:add(nn.TemporalConvolution(1024, 1024, 1, 1))
  model:add(self.nonLinearity())
  model:add(nn.Dropout(0.5))
  -- Final layer
  model:add(nn.TemporalConvolution(1024, structure.nOutputs, 1, 1))
  model:add(nn.Reshape(((structure.nInputs / 8) - 8) * structure.nOutputs))
  model:add(nn.Linear(((structure.nInputs / 8) - 8) * structure.nOutputs, structure.nOutputs))
  return model
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
  self.initialize = nn.kaiming;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = false;
  self.kernelWidth = {};
  self.pretrain = false;
  self.padding = true;
  self.dropout = 0.5;
end

function modelVGG:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end