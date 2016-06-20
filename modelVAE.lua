require 'torch'
require 'nn'
local nninit = require 'nninit'

-------------------------------------
-- 
-- This provides sampling for Gaussian distribution
-- 
-- Input is considered to be
--  input[1] = mean of the distribution
--  input[2] = standard deviation
--  
-------------------------------------
local gaussSampler, parent = torch.class('nn.gaussSampler', 'nn.Module')

-- Initialize sampler
function gaussSampler:__init()
  parent.__init(self)
  self.gradInput = {}
end 

function gaussSampler:updateOutput(input)
  -- Generate random vector
  self.eps = self.eps or input[1].new()
  self.eps:resizeAs(input[1]):copy(torch.randn(input[1]:size()))
  -- Create the output vector
  self.ouput = self.output or self.output.new()
  -- Put the log variance in the vector
  self.output:resizeAs(input[2]):copy(input[2])
  -- Multiply the rand by variance
  self.output:mul(0.5):exp():cmul(self.eps)
  -- Add the mean
  self.output:add(input[1])
  return self.output
end

function gaussSampler:updateGradInput(input, gradOutput)
  -- Gradient of mean is simply replicate
  self.gradInput[1] = self.gradInput[1] or input[1].new()
  self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)
  -- Gradient of the variance
  self.gradInput[2] = self.gradInput[2] or input[2].new()
  self.gradInput[2]:resizeAs(gradOutput):copy(input[2])
  self.gradInput[2]:mul(0.5):exp():mul(0.5):cmul(self.eps)
  self.gradInput[2]:cmul(gradOutput)
  return self.gradInput
end

function gaussSampler:evaluate()
   self.train = false
end

-------------------------------------
-- 
-- This provides the Kullback-Leibler criterion
--  
-------------------------------------
local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

-- Compute the simplified VAE criterion
-- 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
function KLDCriterion:updateOutput(mean, log_var)
  -- Compute square of mean
  local mSquare = torch.pow(mean,2)
  -- Clone the log(sigma^2)  
  local logVar = log_var:clone();
  -- Add the deviation and squared mean
  logVar:exp():mul(-1):add(-1, mSquare);
  -- Add the log and 1
  logVar:add(1):add(logVar);
  -- Compute sum and normalize
  self.output = -0.5 * torch.sum(logVar);
  return self.output
end

function KLDCriterion:updateGradInput(mean, log_var)
  -- Gradient of the input
  self.gradInput = {}
  -- Gradient of mean
  self.gradInput[1] = mean:clone()
  -- Gradient of the variance
  self.gradInput[2] = torch.exp(log_var):mul(-1):add(1):mul(-0.5)
  return self.gradInput
end

-------------------------------------
-- 
-- This provides a Gaussian criterion
--  
-------------------------------------
local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:updateOutput(input, target)
    -- - log(sigma) - 0.5 *(2pi)) - 0.5 * (x - mu)^2/sigma^2
    local Gelement = torch.mul(input[2],0.5):add(0.5 * math.log(2 * math.pi))
    Gelement:add(torch.add(target,-1,input[1]):pow(2):cdiv(torch.exp(input[2])):mul(0.5));
    self.output = torch.sum(Gelement)
    return self.output
end

function GaussianCriterion:updateGradInput(input, target)
    self.gradInput = {}
    -- (x - mu) / sigma^2  --> (1 / sigma^2 = exp(-log(sigma^2)) )
    self.gradInput[1] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1])):mul(-1)
    -- - 0.5 + 0.5 * (x - mu)^2 / sigma^2
    self.gradInput[2] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]):pow(2)):mul(-1):add(0.5)
    return self.gradInput
end

-------------------------------------
-- 
-- Definition of the variational autoencoder
--  
-------------------------------------
local modelVAE, parent = torch.class('modelVAE', 'modelClass')

-- Define a single MLP
function modelVAE:defineSimple(input_size, hidden_layer_size)
  -- Container
  local model = nn.Sequential();
  -- Hidden layers
  model:add(nn.Linear(input_size, hidden_layer_size));
  -- Batch normalization
  if self.batchNormalize then model:add(nn.BatchNormalization(hidden_layer_size)); end
  -- Non-linearity
  model:add(self.nonLinearity());
  -- Dropout
  if self.dropout then model:add(nn.Dropout(self.dropout)); end
  return model;
end

function modelVAE:defineConvolutional(input_size, hidden_layer_size, conv_size, kernel_size, pool_size)
  -- Container:
  local model = nn.Sequential();
  curTPoints = input_size;
  -- Construct convolutional layers
  inS = input_size; inSize = 1; outSize = conv_size;
  model:add(nn.Reshape(input_size, 1));
  if self.padding then model:add(nn.Padding(2, -(kernel_size/2 - 1))); model:add(nn.Padding(2, kernel_size/2)); end
  -- Perform convolution
  model:add(nn.TemporalConvolution(inSize, outSize, kernel_size));
  -- Batch normalization
  if self.batchNormalize then
    model:add(nn.Reshape(curTPoints * outSize)); 
    model:add(nn.BatchNormalization(curTPoints * outSize));
    model:add(nn.Reshape(curTPoints, outSize))
    curTPoints = curTPoints / pool_size;
  end
  -- Non-linearity
  model:add(self.nonLinearity())
  -- Pooling
  model:add(nn.TemporalMaxPooling(structure.poolSize[i], structure.poolSize[i]));
  -- Convolution 
  convOut = conv_size * input_size / pool_size;
  -- And reshape the output of the convolutional layers
  model:add(nn.Reshape(convOut));
  -- Construct final standard layers
  inSize = convOut; outSize = hidden_layer_size;
  -- Linear transform
  model:add(nn.Linear(inSize, outSize));
  -- Batch normalization
  if self.batchNormalize then model:add(nn.BatchNormalization(outSize)); end
  -- Non-linearity
  model:add(self.nonLinearity())
  -- Eventual dropout
  if self.dropout then model:add(nn.Dropout(self.dropout)); end
  return model;
end
-- 
function modelVAE:get_encoder(input_size, hidden_layer_size, latent_variable_size)
     -- Define the encoder base model
    local encoder = self:defineSimple(input_size, hidden_layer_size);
    -- Now separate Gaussian components
    mean_logvar = nn.ConcatTable()
    -- First obtain the mean of the distribution
    mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))
    -- Then obtain the variance of distribution
    mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))
    -- Enclose in the encoder
    encoder:add(mean_logvar)
    return encoder
end

function modelVAE:get_decoder(input_size, hidden_layer_size, latent_variable_size)
  -- The Decoder
  local decoder = self:defineSimple(hidden_layer_size, input_size)
  -- Now separate Gaussian components
  mean_logvar = nn.ConcatTable()
  -- First obtain the mean of the distribution
  mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))
  -- Then obtain the variance of distribution
  mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))
  -- Enclose in the encoder
  decoder:add(mean_logvar);
  return decoder
end

function modelVAE:defineModel(structure, options)
  local model = {};
  -- Create a hierarchy of encoding
  model.encoding = nn.Sequential();
  -- Create a reverse hierarchy of decoding
  model.decoding = nn.Sequential();
  -- Define each layer
  for n = 1,structure.nLayers do
    -- Check the size of current layer
    if n == 1 then nIn = structure.nInputs; nOut = structure.layers[n]; else nIn = structure.layers[n-1]; nOut = structure.layers[n]; end
    -- Create encoding layer
    model.encoding:add(self:get_encoder(nIn, nOut, nOut));
    model.encoding:add(nn.gaussSampler());
    -- Create decoding layer
    model.decoding:insert(nn.gaussSampler(), 1)
    model.decoding:insert(self:get_decoder(nOut, nOut, nIn), 1);
  end
  -- Add a final linear layer
  model.encoding:add(nn.Linear(structure.layers[structure.nLayers],structure.nOutputs));
  --self.model = model
  return model; 
end

function modelVAE:definePretraining(structure, l, options)
  if self.model == nil then self.model = {}; self.model.encoders = {}; self.model.decoders = {}; end
  -- Check the size of current layer
  if l == 1 then nIn = structure.nInputs; nOut = structure.layers[l]; else nIn = structure.layers[l-1]; nOut = structure.layers[l]; end
  -- Create encoding layer
  self.model.encoders[l] = self:get_encoder(nIn, nOut, nOut);
    -- Create decoding layer
  self.model.decoders[l] = self:get_decoder(nOut, nOut, nIn);
  local input = nn.Identity()();
  -- Construct an encoder to retrieve mean and variance
  local mean, log_var = self.model.encoders[l](input):split(2);
  -- Add a sampler on top
  local z = nn.gaussSampler()({mean, log_var});
  local reconstruction, reconstruction_var, model;
  -- Just obtain the reconstruction means and vars for the decoder
  reconstruction, reconstruction_var = self.model.decoders[l](z):split(2);
  -- Construct the complete module
  local model = {};
  model.full = nn.gModule({input},{reconstruction, reconstruction_var, mean, log_var})
  model.encoder = nn.Sequential():add(self.model.encoders[l]):add(nn.gaussSampler());
  self.criterionG = nn.GaussianCriterion();
  self.criterionKL = nn.KLDCriterion();
  return model;
end

function modelVAE:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  return model.encoder;
end

-- Function to perform unsupervised training on a sub-model
function modelVAE:unsupervisedTrain(modelTab, unsupData, options)
  local model = modelTab.full;
  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training();
  -- get all parameters
  parameters,gradParameters = model:getParameters();
  -- training errors
  local err = 0
  local iter = 0
  -- create mini batch
  local inputs = {};
  local targets = {};
  if (unsupData.data[1]:nDimension() == 1) then
    inputs = torch.Tensor(options.batchSize, unsupData.data[1]:size(1))
    targets = torch.Tensor(options.batchSize, unsupData.data[1]:size(1))
  else
    inputs = torch.Tensor(options.batchSize, unsupData.data[1]:size(1), unsupData.data[1]:size(2))
    targets = torch.Tensor(options.batchSize, unsupData.data[1]:size(1), unsupData.data[1]:size(2))
  end
  if options.cuda then inputs = inputs:cuda(); targets = targets:cuda(); end
  for t = 1,math.min(options.maxIter, (unsupData.data:size(1)-options.batchSize)),options.batchSize do
    -- progress
    iter = iter+1
    -- Check size of batch (for last smaller)
    local bSize = math.min(options.batchSize, unsupData.data:size(1) - t + 1);
    local k = 1;
    for i = t,math.min(t+options.batchSize-1,unsupData.data:size(1)) do
      inputs[k] = unsupData.data[i];
      targets[k] = unsupData.data[i];
      k = k + 1;
    end
    local feval = function(x)
      -- Copy current parameters
      if x ~= parameters then parameters:copy(x) end
      -- Reset the gradient parameters
      model:zeroGradParameters()
      -- Retrieve mean and vars (forward pass)
      reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(inputs))
      reconstruction = {reconstruction, reconstruction_var}
      -- Compute the Gaussian part of the criterion
      local err = self.criterionG:forward(reconstruction, inputs)
      -- Compute the Kullback-Leibler part of the criterion
      local KLDerr = self.criterionKL:forward(mean, log_var)
      -- Backward through both criterions
      local df_dw = self.criterionG:backward(reconstruction, inputs)
      local dKLD_dmu, dKLD_dlog_var = unpack(self.criterionKL:backward(mean, log_var))
      error_grads = {df_dw[1], df_dw[2], dKLD_dmu, dKLD_dlog_var}
      -- Backward through the model
      model:backward(inputs, error_grads)
      local batchlowerbound = err + KLDerr
      return batchlowerbound, gradParameters
    end
    -- optimize on current mini-batch
    _,fs = optimMethod(feval, parameters, optimState)
    -- Make error independent of batch size
    err = err + fs[1] * options.batchSize
  end
  return err;
end

-- Function to perform unsupervised testing on a sub-model
function modelVAE:unsupervisedTest(modelTab, data, options)
  local model = modelTab.full;
  inputs = data.data;
  reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(inputs))
  reconstruction = {reconstruction, reconstruction_var}
  -- Compute the Gaussian part of the criterion
  local err = self.criterionG:forward(reconstruction, inputs)
  -- Compute the Kullback-Leibler part of the criterion
  local KLDerr = self.criterionKL:forward(mean, log_var)
  return err + KLDerr;
end

function modelVAE:getParameters(model)
  return model.encoding:getParameters();
end

-- Function to perform supervised training on the full model
function modelVAE:supervisedTrain(model, data, options)
  return supervisedTrain(model.encoding, data, options);
end

-- Function to perform supervised testing on the model
function modelVAE:supervisedTest(model, data, options)
  return supervisedTest(model.encoding, data, options);
end

function modelVAE:defineCriterion(model)
  model.encoding:add(nn.LogSoftMax());
  criterion = nn.ClassNLLCriterion();
  return model, criterion;
end

function modelVAE:weightsInitialize(model)
  -- TODO
  return model;
end

function modelVAE:weightsTransfer(model, trainedLayers)
  -- TODO
  return model;
end

function modelVAE:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.pretrainType = 'ae';
  self.pretrain = true;
  self.dropout = 0.5;
  self.layerWise = true;
end

function modelVAE:parametersRandom()
  -- All possible non-linearities
  self.distributions = {};
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
  self.distributions.batchNormalize = {true, false};
  self.distributions.pretrainType = {'ae', 'psd'};
  self.distributions.pretrain = {true, false};
  self.distributions.dropout = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
end

--[[

-- Joost van Amersfoort - <joost@joo.st>
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

nngraph.setDebug(false)

local VAE = require 'VAE'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'

--For loading data files
require 'load'

local continuous = false
data = load(continuous)

local input_size = data.train:size(2)
local latent_variable_size = 20
local hidden_layer_size = 400

local batch_size = 100

torch.manualSeed(1)

local encoder = VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
local decoder = VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})

local reconstruction, reconstruction_var, model
if continuous then
    reconstruction, reconstruction_var = decoder(z):split(2)
    model = nn.gModule({input},{reconstruction, reconstruction_var, mean, log_var})
    criterion = nn.GaussianCriterion()
else
    reconstruction = decoder(z)
    model = nn.gModule({input},{reconstruction, mean, log_var})
    criterion = nn.BCECriterion()
    criterion.sizeAverage = false
end

-- Some code to draw computational graph
-- dummy_x = torch.rand(dim_input)
-- model:forward({dummy_x})

-- Uncomment to get structure of the Variational Autoencoder
-- graph.dot(.fg, 'Variational Autoencoder', 'VA')

KLD = nn.KLDCriterion()

local parameters, gradients = model:getParameters()

local config = {
    learningRate = 0.001
}

local state = {}

epoch = 0
while true do
    epoch = epoch + 1
    local lowerbound = 0
    local tic = torch.tic()

    local shuffle = torch.randperm(data.train:size(1))

    -- This batch creation is inspired by szagoruyko CIFAR example.
    local indices = torch.randperm(data.train:size(1)):long():split(batch_size)
    indices[#indices] = nil
    local N = #indices * batch_size

    local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        local inputs = data.train:index(1,v)

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            model:zeroGradParameters()
            local reconstruction, reconstruction_var, mean, log_var
            if continuous then
                reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(inputs))
                reconstruction = {reconstruction, reconstruction_var}
            else
                reconstruction, mean, log_var = unpack(model:forward(inputs))
            end

            local err = criterion:forward(reconstruction, inputs)
            local df_dw = criterion:backward(reconstruction, inputs)

            local KLDerr = KLD:forward(mean, log_var)
            local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))

            if continuous then
                error_grads = {df_dw[1], df_dw[2], dKLD_dmu, dKLD_dlog_var}
            else
                error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
            end

            model:backward(inputs, error_grads)

            local batchlowerbound = err + KLDerr

            return batchlowerbound, gradients
        end

        x, batchlowerbound = optim.adam(opfunc, parameters, config, state)

        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("Epoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. torch.toc(tic)) 

    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
    end

    if epoch % 2 == 0 then
        torch.save('save/parameters.t7', parameters)
        torch.save('save/state.t7', state)
        torch.save('save/lowerbound.t7', torch.Tensor(lowerboundlist))
    end
end

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(mean, log_var)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()

    KLDelements:exp():mul(-1)
    KLDelements:add(-1, mean_sq)
    KLDelements:add(1)
    KLDelements:add(log_var)

    self.output = -0.5 * torch.sum(KLDelements)

    return self.output
end

function KLDCriterion:updateGradInput(mean, log_var)
  self.gradInput = {}

    self.gradInput[1] = mean:clone()

    -- Fix this to be nicer
    self.gradInput[2] = torch.exp(log_var):mul(-1):add(1):mul(-0.5)

    return self.gradInput
end

require 'nn'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:updateOutput(input, target)
    -- - log(sigma) - 0.5 *(2pi)) - 0.5 * (x - mu)^2/sigma^2
    -- input[1] = mu
    -- input[2] = log(sigma^2)

    local Gelement = torch.mul(input[2],0.5):add(0.5 * math.log(2 * math.pi))
    Gelement:add(torch.add(target,-1,input[1]):pow(2):cdiv(torch.exp(input[2])):mul(0.5))

    self.output = torch.sum(Gelement)

    return self.output
end

function GaussianCriterion:updateGradInput(input, target)
    self.gradInput = {}

    -- (x - mu) / sigma^2  --> (1 / sigma^2 = exp(-log(sigma^2)) )
    self.gradInput[1] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1])):mul(-1)

    -- - 0.5 + 0.5 * (x - mu)^2 / sigma^2
    self.gradInput[2] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]):pow(2)):mul(-1):add(0.5)

    return self.gradInput
end

--]]