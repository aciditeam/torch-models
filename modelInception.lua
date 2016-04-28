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
require 'torch'
local nninit = require 'nninit'

----------------------------------------------------------------------
-- Temporal average pooling layer
----------------------------------------------------------------------
local TemporalAveragePooling, Parent = torch.class('nn.MyTemporalAveragePooling', 'nn.TemporalSubSampling')

function TemporalAveragePooling:__init(nInputPlane, kW, dW)
   Parent.__init(self, nInputPlane, kW, dW)
end

function TemporalAveragePooling:reset()
   self.weight:fill(1.0)
   self.bias:fill(0.0)
end
-- avoid parameter update
function TemporalAveragePooling:accGradParameters()
end
function TemporalAveragePooling:accUpdateGradParameters()
end
function TemporalAveragePooling:updateParameters()
end

----------------------------------------------------------------------
-- Spatial average pooling layer
----------------------------------------------------------------------
local SpatialAveragePooling, Parent = torch.class('nn.MySpatialAveragePooling', 'nn.SpatialSubSampling')

function SpatialAveragePooling:__init(nInputPlane, kW, kH, dW, dH)
   Parent.__init(self, nInputPlane, kW, kH, dW, dH)
end

function SpatialAveragePooling:reset()
   self.weight:fill(1.0)
   self.bias:fill(0.0)
end
-- avoid parameter update
function SpatialAveragePooling:accGradParameters()
end
function SpatialAveragePooling:accUpdateGradParameters()
end
function SpatialAveragePooling:updateParameters()
end

local modelInception, parent = torch.class('modelInception', 'modelCNN')

----------------------------------------------------------------------
-- A single temporal inception module
-- (From the Oxford course)
----------------------------------------------------------------------
function modelInception:temporalInception(depth_dim, input_size, config, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  local conv1 = nil   
  local conv3 = nil
  local conv5 = nil
  local pool = nil
  -- Define depth concatenation
  local depth_concat = nn.DepthConcat(depth_dim);
  -- First sub-convolution module
  conv1 = nn.Sequential();
  conv1:add(nn.TemporalConvolution(input_size, config[1][1], 1, 1));
  if self.batchNormalize then
    conv1:add(nn.Reshape(config[1][2] * config[1][1])); 
    conv1:add(nn.BatchNormalization(config[1][2] * config[1][1]));
    conv1:add(nn.Reshape(config[1][2], config[1][1])); 
  end
  conv1:add(self.nonLinearity());
  if self.dropout then conv1:add(nn.Dropout(self.dropout)) end
  -- Second sub-convolution module
  depth_concat:add(conv1);
  conv3 = nn.Sequential();
  conv3:add(nn.TemporalConvolution(input_size, config[2][1], 1, 1));
  if self.batchNormalize then
    conv3:add(nn.Reshape(config[1][2] * config[2][1])); 
    conv3:add(nn.BatchNormalization(config[1][2] * config[2][1]));
    conv3:add(nn.Reshape(config[1][2], config[2][1])); 
  end
  conv3:add(self.nonLinearity());
  if self.dropout then conv3:add(nn.Dropout(self.dropout)) end
  if self.padding then conv3:add(nn.Padding(2, -1)); conv3:add(nn.Padding(2, 1)); end
  conv3:add(nn.TemporalConvolution(config[2][1], config[2][2], 3, 3))
  conv3:add(self.nonLinearity());
  if self.dropout then conv3:add(nn.Dropout(self.dropout)) end
  depth_concat:add(conv3);
  -- Third sub-convolution module
  conv5 = nn.Sequential();
  conv5:add(nn.TemporalConvolution(input_size, config[3][1], 1, 1))
  if self.batchNormalize then
    conv5:add(nn.Reshape(config[1][2] * config[3][1])); 
    conv5:add(nn.BatchNormalization(config[1][2] * config[3][1]));
    conv5:add(nn.Reshape(config[1][2], config[3][1])); 
  end
  conv5:add(self.nonLinearity());
  if self.dropout then conv5:add(nn.Dropout(self.dropout)) end
  if self.padding then conv5:add(nn.Padding(2, -2)); conv5:add(nn.Padding(2, 2)); end
  conv5:add(nn.TemporalConvolution(config[3][1], config[3][2], 5, 5))
  conv5:add(self.nonLinearity());
  if self.dropout then conv5:add(nn.Dropout(self.dropout)) end
  depth_concat:add(conv5);
  -- Pooling layer
  pool = nn.Sequential()
  pool:add(nn.TemporalMaxPooling(config[4][1], config[4][1]))
  pool:add(nn.TemporalConvolution(input_size, config[4][2], 1, 1))
  --if self.batchNormalize then
  --  pool:add(nn.Reshape(input_size * config[4][2])); 
  --  pool:add(nn.BatchNormalization(input_size * config[4][2]));
  --  pool:add(nn.Reshape(input_size, config[4][2])); 
  --end
  pool:add(self.nonLinearity());
  if self.dropout then pool:add(nn.Dropout(self.dropout)) end
  depth_concat:add(pool)
  return depth_concat
end

function modelInception:defineModel(structure, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  local model = nn.Sequential() 
  -- Reshape the input to fit the convolution
  model:add(nn.Reshape(structure.nInputs, 1));
  model:add(nn.Padding(2, -1));
  model:add(nn.Padding(2, 1)); 
   -- First convolution layer (VGG configuration)
  model:add(nn.TemporalConvolution(1, 64, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.TemporalConvolution(64, 64, 3, 1))
  model:add(self.nonLinearity())
  model:add(nn.Padding(2, -1));
  model:add(nn.Padding(2, 1)); 
  model:add(nn.TemporalMaxPooling(2, 2))
  -- Inception 3a
  model:add(self:temporalInception(3, 64, {{64, 64}, {96, 128}, {16, 32}, {3, 32}}, options))
  -- Inception 3b
  model:add(self:temporalInception(3, 256, {{128, 64}, {128, 192}, {32, 96}, {3, 64}}, options))
  -- Max-pooling layer
  model:add(nn.TemporalMaxPooling(2, 2))
  -- Inception 4a
  model:add(self:temporalInception(3, 480, {{192, 32}, {96, 208}, {16, 48}, {3, 64}}, options))
  -- Inception 4b
  model:add(self:temporalInception(3, 512, {{160, 32}, {112, 224}, {24, 64}, {3, 64}}, options))
  -- Inception 4c
  model:add(self:temporalInception(3, 512, {{128, 32}, {128, 256}, {24, 64}, {3, 64}}, options))
  -- Inception 4d
  --model:add(inception_module(3, 512, {{112}, {144, 288}, {32, 64}, {3, 64}}, options))
  -- Inception 4e
  --model:add(inception_module(3, 528, {{256}, {160, 320}, {32, 128}, {3, 128}}, options))   
  -- Global average pooling
  model:add(nn.TemporalMaxPooling(6, 6));
  model:add(nn.Dropout(0.4))
  model:add(nn.TemporalConvolution(512, structure.nOutputs, 1, 1))
  model:add(nn.Reshape(torch.round(structure.nInputs / 24.5) * structure.nOutputs));
  model:add(nn.Linear(torch.round(structure.nInputs / 24.5) * structure.nOutputs, structure.nOutputs));
  return model
end

function modelCNN:defineInceptionPretraining(structure, l, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- Get the fixed structure
  local fixedStructure = self:defineStructure();
  --[[ Encoder part ]]--
  encoder = self:temporalInception(3, fixedStructure.inputSizes[l - 1], fixedStructure.inceptions[l - 1], options)
  -- Decoder:
  decoder = nn.Sequential()
  -- Put de-convolution
  curTPoints = structure.nInputs
  for i = 1,l do curTPoints = curTPoints / fixedStructure.poolSize[i]; end
  decoder:add(nn.Reshape(curTPoints * fixedStructure.inputSizes[l]));
  decoder:add(nn.Linear(curTPoints * fixedStructure.inputSizes[l], fixedStructure.inputSizes[l - 1] * curTPoints))
  decoder:add(nn.Reshape(fixedStructure.inputSizes[l - 1], curTPoints))
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, options.beta)
  return model;
end

function modelCNN:defineVGGPretraining(structure, l, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- Get the fixed structure
  local fixedStructure = self:defineStructure();
  --[[ Encoder part ]]--
  encoder = nn.Sequential()
  -- Prepare the layer properties
  inS = 1; inSize = structure.nInputs;
  encoder:add(nn.Reshape(structure.nInputs, 1));
  outS = structure.convSize[l]; 
  -- Eventual padding
  encoder:add(nn.Padding(2, -1)); encoder:add(nn.Padding(2, 1));
  -- Perform convolution
  encoder:add(nn.TemporalConvolution(inS, fixedStructure.vggSizes[1], fixedStructure.kernelWidth[1]));
  -- Batch normalization
  --if self.batchNormalize then
  --  curTPoints = structure.nInputs
  --  for i = 2,l do curTPoints = curTPoints / structure.poolSize[i]; end
  --  encoder:add(nn.Reshape(curTPoints * outS)); 
  --  encoder:add(nn.BatchNormalization(curTPoints * outS));
  --  encoder:add(nn.Reshape(curTPoints, outS))
  --end
  -- Non-linearity
  encoder:add(self.nonLinearity());
  -- Perform convolution
  encoder:add(nn.TemporalConvolution(fixedStructure.vggSizes[1], fixedStructure.vggSizes[2], fixedStructure.kernelWidth[2]));
  -- Non-linearity
  encoder:add(self.nonLinearity()); 
  -- Padding
  encoder:add(nn.Padding(2, -1)); encoder:add(nn.Padding(2, 1));
  -- Pooling
  encoder:add(nn.TemporalMaxPooling(fixedStructure.poolSize[1], fixedStructure.poolSize[1]));
  -- Decoder:
  decoder = nn.Sequential()
  -- Put de-convolution
  decoder:add(nn.Reshape((structure.nInputs / fixedStructure.poolSize[1]) * fixedStructure.vggSizes[2]))
  decoder:add(nn.Linear((structure.nInputs / fixedStructure.poolSize[1]) * fixedStructure.vggSizes[2], structure.nInputs))
  decoder:add(nn.Reshape(structure.nInputs, 1))
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, options.beta)
  return model;
end

function modelInception:definePretraining(structure, l, options)
  local model = {};
  if l == 1 then
    model = self:defineVGGPretraining(structure, l, options);
  else
    model = self:defineInceptionPretraining(structure, l, options);
  end
  return model;
end

function modelInception:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder;
  --encoder:remove();
  return encoder;
end

function modelInception:defineStructure()
  -- Create a pre-defined structure
  local structure = {};
  -- Properties of the convolutional network
  structure.nLayers       = 5
  structure.nInceptions   = 5
  structure.inceptions    = {{{64, 64}, {96, 128}, {16, 32}, {3, 32}},
                            {{128, 64}, {128, 192}, {32, 96}, {3, 64}},
                            {{192, 32}, {96, 208}, {16, 48}, {3, 64}},
                            {{160, 32}, {112, 224}, {24, 64}, {3, 64}},
                            {{128, 32}, {128, 256}, {24, 64}, {3, 64}},
                            {{112}, {144, 288}, {32, 64}, {3, 64}},
                            {{256}, {160, 320}, {32, 128}, {3, 128}}}; 
  structure.inputSizes    = {64, 256, 480, 512, 512, 512, 528}; 
  structure.vggSizes      = {64, 64}; 
  structure.kernelWidth   = {3,3}
  structure.kernelStep    = {2,2,1,1,1}
  structure.poolSize      = {2,1,1,2,1,1,1}
  structure.padding       = {true,false,true,true,true}
  structure.normalize     = {true,true,false,false,false}
  structure.pool          = {true,true,false,false,false}
  return structure;
end

function modelInception:weightsInitialize(model)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', self.initialize);
    module:init('bias', self.initialize);
  end
  -- Do the same for convolutional modules
  convNodes = model:findModules('nn.TemporalConvolution')
  for l = 1,#convNodes do
    module = convNodes[l];
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

function modelInception:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.RReLU;
  self.batchNormalize = true;
  self.kernelWidth = {};
  self.pretrain = false;
  self.padding = true;
  self.dropout = 0.5;
end

function modelInception:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end

----------------------------------------------------------------------
-- Spatial inception
-- (Kept just in case)
----------------------------------------------------------------------
function spatialInception(depth_dim, input_size, config)
   local conv1 = nil   
   local conv3 = nil
   local conv5 = nil
   local pool = nil
   local depth_concat = nn.DepthConcat(depth_dim)
   conv1 = nn.Sequential()
   conv1:add(nn.SpatialConvolutionMM(input_size, config[1][1], 1, 1))
   conv1:add(nn.ReLU())
   depth_concat:add(conv1)
   conv3 = nn.Sequential()
   conv3:add(nn.SpatialConvolutionMM(input_size, config[2][1], 1, 1))
   conv3:add(nn.ReLU())
   conv3:add(nn.SpatialConvolutionMM(config[2][1], config[2][2], 3, 3))
   conv3:add(nn.ReLU())
   depth_concat:add(conv3)
   conv5 = nn.Sequential()
   conv5:add(nn.SpatialConvolutionMM(input_size, config[3][1], 1, 1))
   conv5:add(nn.ReLU())
   conv5:add(nn.SpatialConvolutionMM(config[3][1], config[3][2], 5, 5))
   conv5:add(nn.ReLU())
   depth_concat:add(conv5)
   pool = nn.Sequential()
   pool:add(nn.SpatialMaxPooling(config[4][1], config[4][1], 1, 1))
   pool:add(nn.SpatialConvolutionMM(input_size, config[4][2], 1, 1))
   pool:add(nn.ReLU())
   depth_concat:add(pool)
   return depth_concat
end

function modelInception:defineSpatial(structure, options) -- validate.lua Acc:
   local model = nn.Sequential() 
   -- first convolution layer (VGG configuration)
   model:add(nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- inception 3a
   model:add(inception_module(2, 64, {{64}, {96, 128}, {16, 32}, {3, 32}}))
   -- inception 3b
   model:add(inception_module(2, 256, {{128}, {128, 192}, {32, 96}, {3, 64}}))
   -- maxpool
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- inception 4a
   model:add(inception_module(2, 480, {{192}, {96, 208}, {16, 48}, {3, 64}}))
   -- inception 4b
   model:add(inception_module(2, 512, {{160}, {112, 224}, {24, 64}, {3, 64}}))
   -- inception 4c
   model:add(inception_module(2, 512, {{128}, {128, 256}, {24, 64}, {3, 64}}))
   -- inception 4d
   --model:add(inception_module(2, 512, {{112}, {144, 288}, {32, 64}, {3, 64}}))
   -- inception 4e
   --model:add(inception_module(2, 528, {{256}, {160, 320}, {32, 128}, {3, 128}}))
   -- global avgpool
   model:add(nn.MySpatialAveragePooling(512, 6, 6, 6, 6))
   model:add(nn.Dropout(0.4))
   model:add(nn.SpatialConvolutionMM(512, 10, 1, 1, 1, 1))
   model:add(nn.Reshape(10))
   model:add(nn.SoftMax())
   return model
end
