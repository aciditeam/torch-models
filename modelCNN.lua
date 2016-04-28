----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Convolutional Neural Network
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'nn'
require 'torch'
require 'nninit'
require 'modelClass'
local nninit = require 'nninit'

local modelCNN, parent = torch.class('modelCNN', 'modelClass')

function modelCNN:defineModel(structure, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- Container:
  model = nn.Sequential();
  curTPoints = structure.nInputs;
  -- Construct convolutional layers
  for i = 1,structure.nLayers do
    -- Reshape inputs
    if i == 1 then
      inS = structure.nInputs; inSize = 1; outSize = structure.convSize[i];
      model:add(nn.Reshape(structure.nInputs, 1));
    else
      inSize = structure.convSize[i-1]; inS = inSize; outSize = structure.convSize[i];
    end
    -- Eventual padding ?
    if self.padding then model:add(nn.Padding(2, -(structure.kernelWidth[i]/2 - 1))); model:add(nn.Padding(2, structure.kernelWidth[i]/2)); end
    -- Perform convolution
    model:add(nn.TemporalConvolution(inSize, outSize, structure.kernelWidth[i]));
    -- Batch normalization
    if self.batchNormalize then
      model:add(nn.Reshape(curTPoints * outSize)); 
      model:add(nn.BatchNormalization(curTPoints * outSize));
      model:add(nn.Reshape(curTPoints, outSize))
      curTPoints = curTPoints / structure.poolSize[i];
    end
    -- Non-linearity
    model:add(self.nonLinearity())
    -- Pooling
    model:add(nn.TemporalMaxPooling(structure.poolSize[i], structure.poolSize[i]));
  end
  convOut = structure.convSize[#structure.convSize] * (structure.nInputs / torch.Tensor(structure.poolSize):cumprod()[#structure.poolSize]);
  -- Keep the first kernel width for pre-training
  self.kernelWidth = structure.kernelWidth[1];
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

function modelCNN:definePretraining(structure, l, options)
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

function modelCNN:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder;
  --encoder:remove();
  return encoder;
end

function modelCNN:weightsInitialize(model)
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

function modelCNN:weightsTransfer(model, trainedLayers)
  -- TODO
  -- What about classifying layers in pre-training ?
  -- TODO
  -- Find linear modules (also inside convolutional layers)
  linearNodes = model:findModules('nn.Linear');
  -- Find only the convolutional modules
  convNodes = model:findModules('nn.TemporalConvolution')
  -- Current linear layer
  local curLayer = 1;
  local curConv = 1;
  for l = 1,#trainedLayers do
    -- Find equivalent in pre-trained layer
    linNodes = trainedLayers[l].encoder:findModules('nn.Linear');
    for k = 1,#linNodes do
      linearNodes[curLayer].weight = linNodes[k].weight;
      linearNodes[curLayer].bias = linNodes[k].bias;
      curLayer = curLayer + 1;
    end
    -- Find equivalent in pre-trained layer
    preNodes = trainedLayers[l].encoder:findModules('nn.TemporalConvolution');
    for k = 1,#preNodes do
      convNodes[curConv].weight = preNodes[k].weight;
      convNodes[curConv].bias = preNodes[k].bias;
      curConv = curConv + 1;
    end
  end
  return model;
end

function modelCNN:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.kernelWidth = {};
  self.pretrain = true;
  self.padding = true;
  self.dropout = 0.5;
end

function modelCNN:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end