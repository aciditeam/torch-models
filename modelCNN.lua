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

-- TODO
-- If cuda, then need to switch between
-- * nn vs cunn
-- * cudnn ... (no temporal)
-- * ccn2 ?
-- TODO

local modelCNN, parent = torch.class('modelCNN', 'modelClass')

function modelCNN:defineModel(structure, options)
  -- Container:
  model = nn.Sequential();
  -- Construct convolutional layers
  for i = 1,structure.nLayers do
    -- Reshape inputs
    if i == 1 then
      inSize = 1; outSize = structure.convSize[i];
      model:add(nn.Reshape(structure.nInputs, 1));
    else
      inSize = structure.convSize[i-1]; outSize = structure.convSize[i];
    end
    -- Eventual padding ?
    if self.padding then model:add(nn.Padding(1, -(structure.kernelWidth[i]/2 - 1))); model:add(nn.Padding(1, structure.kernelWidth[i]/2)); end
    -- Perform convolution
    model:add(nn.TemporalConvolution(inSize, outSize, structure.kernelWidth[i]));
    -- Batch normalization
    if self.batchNormalize then encoder:add(nn.BatchNormalization(outS)); end
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
    if self.batchNormalize then encoder:add(nn.BatchNormalization(outS)); end
    -- Non-linearity
    model:add(self.nonLinearity())
    -- Eventual dropout
    if self.dropout then model:add(nn.Dropout(self.dropout)); end
  end
  model:add(nn.Linear(structure.layers[structure.nClassLayers], structure.nOutputs));
  return model;
end

function modelCNN:definePretraining(inS, outS, options)
  --[[ Encoder part ]]--
  -- params:
  local kW = options.kernelSize
  -- encoder:
  encoder = nn.Sequential()
  -- Eventual reshape of input
  encoder:add(nn.Reshape(inS, 1));
  -- Eventual padding
  if options.padding then model:add(nn.Padding(1, -kW/2)); model:add(nn.Padding(1, kW/2)); end
  -- Perform convolution
  encoder:add(nn.TemporalConvolution(inS, outS, options.kernelWidth));
  -- Batch normalization
  if self.batchNormalize then encoder:add(nn.BatchNormalization(outS)); end
  -- Non-linearity
  encoder:add(options.nonLinearity())
  -- Put diag
  encoder:add(nn.Diag(outS));
  -- Decoder:
  decoder = nn.Sequential()
  -- Put de-convolution
  decoder:add(nn.TemporalConvolution(outS, inS, options.kernelWidth))
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, params.beta)
  -- impose weight sharing
  -- decoder:get(1).weight = encoder:get(1).weight:t();
  -- decoder:get(1).gradWeight = encoder:get(1).gradWeight:t();
  return model;
end

function modelCNN:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  return model.encoder;
end

function modelCNN:weightsInitialize(model)
  --[[
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
  ]]--
  return model;
end

function modelCNN:weightsTransfer(model, trainedLayers)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,trainedLayers do
    -- Find equivalent in pre-trained layer
    preTrained = trainedLayers[l]:findModules('nn.Linear');
    linearNodes[l].weight = preTrained[1].weight;
    linearNodes[l].bias = preTrained[1].bias;
  end
  -- Find only the convolutional modules
  convNodes = model:findModules('nn.TemporalConvolution')
  for l = 1,trainedLayers do
    -- Find equivalent in pre-trained layer
    preTrained = trainedLayers[l]:findModules('nn.TemporalConvolution');
    convNodes[l].weight = preTrained[1].weight;
    convNodes[l].bias = preTrained[1].bias;
  end
  return model;
end

function modelCNN:parametersDefault()
  self.initialize = nn.kaiming;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = false;
  self.kernelWidth = {};
  self.pretrain = false;
  self.padding = true;
  self.dropout = 0.5;
end

function modelCNN:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end