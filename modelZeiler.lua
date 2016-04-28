----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Convolutional Neural Network - Zeiler network
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'nn'
require 'torch'
require 'nninit'
require 'modelClass'
local nninit = require 'nninit'

local modelZeiler, parent = torch.class('modelZeiler', 'modelClass')
  
function modelZeiler:defineModel(structure, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- Retrieve the fixed structure
  local fixedStructure = self:defineStructure();
  -- Container:
  model = nn.Sequential();
  curTPoints = structure.nInputs;
  -- Construct convolutional layers
  for i = 1,fixedStructure.nLayers do
    -- Reshape inputs
    if i == 1 then
      inS = structure.nInputs; inSize = 1; outSize = fixedStructure.convSize[i];
      model:add(nn.Reshape(structure.nInputs, 1));
    else
      inSize = fixedStructure.convSize[i-1]; inS = inSize; outSize = fixedStructure.convSize[i];
    end
    -- Eventual padding ?
    if fixedStructure.padding[i] then model:add(nn.Padding(2, -(fixedStructure.kernelWidth[i]/2 - 1))); model:add(nn.Padding(2, fixedStructure.kernelWidth[i]/2)); end
    -- Perform convolution
    model:add(nn.TemporalConvolution(inSize, outSize, fixedStructure.kernelWidth[i], fixedStructure.kernelStep[i]));
    -- Batch normalization
    if self.batchNormalize then
      curTPoints = curTPoints / fixedStructure.kernelStep[i];
      model:add(nn.Reshape(curTPoints * outSize)); 
      model:add(nn.BatchNormalization(curTPoints * outSize));
      model:add(nn.Reshape(curTPoints, outSize))
      curTPoints = curTPoints / fixedStructure.poolSize[i];
    end
    -- Non-linearity
    model:add(self.nonLinearity())
    -- Pooling
    if fixedStructure.pooling then model:add(nn.TemporalMaxPooling(fixedStructure.poolSize[i], fixedStructure.poolSize[i])); end
  end
  -- Compute size of convolutional output
  convOut = fixedStructure.convSize[#fixedStructure.convSize] * (structure.nInputs / torch.Tensor(fixedStructure.poolSize):cumprod()[#fixedStructure.poolSize]);
  -- Keep the first kernel width for pre-training
  self.kernelWidth = fixedStructure.kernelWidth[1];
  -- And reshape the output of the convolutional layers
  model:add(nn.Reshape(convOut));
  -- Construct final standard layers
  for i = 1,structure.nClassLayers do
    if i == 1 then
      inSize = convOut; outSize = structure.layers[i];
    else
      inSize = structure.layers[i-1]; outSize = structure.layers[i];
    end
    -- Linear transform
    model:add(nn.Linear(inSize, outSize));
    -- Batch normalization
    if self.batchNormalize then model:add(nn.BatchNormalization(outSize)); end
    -- Non-linearity
    model:add(self.nonLinearity())
    -- Eventual dropout
    if self.dropout then model:add(nn.Dropout(self.dropout)); end
  end
  model:add(nn.Linear(structure.layers[structure.nClassLayers], structure.nOutputs));
  return model;
end

function modelZeiler:defineStructure()
  -- Create a pre-defined structure
  local structure = {};
  -- Properties of the convolutional network
  structure.nLayers       = 5
  structure.convSize      = {96, 256, 384, 384, 256}
  structure.kernelWidth   = {7,5,3,3,3}
  structure.kernelStep    = {2,2,1,1,1}
  structure.poolSize      = {3,3,1,3,3}
  structure.padding       = {true,false,true,true,true}
  structure.normalize     = {true,true,false,false,false}
  structure.pool          = {true,true,false,false,false}
  return structure;
end

function modelZeiler:definePretraining(structure, l, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  --[[ Encoder part ]]--
  encoder = nn.Sequential()
  -- Prepare the layer properties
  if l == 1 then 
    inS = 1; inSize = structure.nInputs;
    encoder:add(nn.Reshape(structure.nInputs, 1));
  else 
    inS = structure.convSize[l - 1]; 
    inSize = inS;
  end
  outS = structure.convSize[l]; 
  -- Eventual padding
  if self.padding then encoder:add(nn.Padding(2, -(structure.kernelWidth[l]/2 - 1))); encoder:add(nn.Padding(2, structure.kernelWidth[l]/2)); end
  -- Perform convolution
  encoder:add(nn.TemporalConvolution(inS, outS, structure.kernelWidth[l]));
  -- Batch normalization
  if self.batchNormalize then
    curTPoints = structure.nInputs
    for i = 2,l do curTPoints = curTPoints / structure.poolSize[i]; end
    encoder:add(nn.Reshape(curTPoints * outS)); 
    encoder:add(nn.BatchNormalization(curTPoints * outS));
    encoder:add(nn.Reshape(curTPoints, outS))
  end
  -- Non-linearity
  encoder:add(self.nonLinearity());
  -- Pooling
  encoder:add(nn.TemporalMaxPooling(structure.poolSize[l], structure.poolSize[l]));
  -- Decoder:
  decoder = nn.Sequential()
  -- Put de-convolution
  curTPoints = structure.nInputs
  for i = 1,l do curTPoints = curTPoints / structure.poolSize[i]; end
  outTPoints = (l == 1) and 1 or structure.nInputs;
  for i = 2,l do outTPoints = outTPoints / structure.poolSize[i]; end
  decoder:add(nn.Reshape(curTPoints * outS))
  decoder:add(nn.Linear(curTPoints * outS, outTPoints * inSize))
  decoder:add(nn.Reshape(outTPoints, inSize))
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, options.beta)
  return model;
end

function modelZeiler:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder;
  --encoder:remove();
  return encoder;
end

function modelZeiler:weightsInitialize(model)
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
  return model;
end

function modelZeiler:weightsTransfer(model, trainedLayers)
  -- TODO
  --
  -- What about classifying layers in pre-training ?
  --
  -- TODO
  --[[ Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#linearNodes do
    -- Find equivalent in pre-trained layer
    preTrained = trainedLayers[l]:findModules('nn.Linear');
    linearNodes[l].weight = preTrained[1].weight;
    linearNodes[l].bias = preTrained[1].bias;
  end
  ]]--
  -- Find only the convolutional modules
  convNodes = model:findModules('nn.TemporalConvolution')
  for l = 1,#convNodes do
    -- Find equivalent in pre-trained layer
    preTrained = trainedLayers[l].encoder:findModules('nn.TemporalConvolution');
    convNodes[l].weight = preTrained[1].weight;
    convNodes[l].bias = preTrained[1].bias;
  end
  return model;
end

function modelZeiler:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.RReLU;
  self.batchNormalize = true;
  self.kernelWidth = {};
  self.pretrain = false;
  self.padding = true;
  self.dropout = 0.5;
end

function modelZeiler:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end