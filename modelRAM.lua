----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'rnn'
require 'unsup'
require 'optim'
require 'torch'
local nninit = require 'nninit'

modelRAM = {};

------------------------------------------------------------------------
--[[ TemporalGlimpse ]]--
-- Similarly to spatial glimpses, a temporal glimpse is the concatenation 
-- of down-scaled parts of a temporal process with increasing scale around 
-- a given location in a time series.
-- input is a pair of Tensors: {timeseries, location}
-- locations are x,y coordinates of the center of cropped patches. 
-- Coordinates are between -1,-1 (top-left) and 1,1 (bottom right)
-- output is a batch of glimpses taken in image at location (x,y)
-- glimpse size is {height, width}, or width only if square-shaped
-- depth is number of patches to crop per glimpse (one patch per scale)
-- Each successive patch is scale x size of the previous patch
------------------------------------------------------------------------
local TemporalGlimpse, parent = torch.class("nn.TemporalGlimpse", "nn.Module")

----------------------------------------------------------------------
-- A real resampling function for time series
----------------------------------------------------------------------
local function tensorResampling(data, destSize, type)
  -- Set the type of kernel
  local type = type or 'gaussian'
  -- Check properties of input data
  if data:dim() == 1 then
    data:resize(1, data:size(1));
  end
  -- Original size of input
  inSize = data:size(2);
  -- Construct a temporal convolution object
  interpolator = nn.TemporalConvolution(inSize, destSize, 1, 1);
  -- Zero-out the whole weights
  interpolator.weight:zeros(destSize, inSize);
  -- Lay down a set of kernels
  for i = 1, destSize do
    if type == 'gaussian' then
      interpolator.weight[i] = image.gaussian1D(inSize, (1 / inSize), 1, true, i / destSize);
    else
      -- No handling of boundaries right now
      for j = math.max({i-kernSize, 1}),math.min({i+kernSize,destSize}) do
        -- Current position in kernel
        relIdx = (j - i) / kernSize;
        if type == 'bilinear' then
          interpolator.weight[i][j] = 1 - math.abs(relIdx);
        elseif type == 'hermite' then
          interpolator.weight[i][j] = (2 * (math.abs(x) ^ 3)) - (3 * (math.abs(x) ^ 2)) + 1;
        elseif type == 'lanczos' then
          interpolator.weight[i][j] = (2 * (math.abs(x) ^ 3)) - (3 * (math.abs(x) ^ 2)) + 1;
        end
      end
    end
  end
  -- print(interpolator.weight);
  return interpolator:forward(data);
end

--function TemporalGlimpse:forget()
--  print('Trying to forget recurrent interaction')
--end

function TemporalGlimpse:__init(size, depth, scale)
   require 'nnx'
   -- Keep inputs
   self.size = size
   self.depth = depth or 3
   self.scale = scale or 2
   -- Check validity
   assert(torch.type(self.size) == 'number')
   assert(torch.type(self.depth) == 'number')
   assert(torch.type(self.scale) == 'number')
   -- Perform init
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   -- Resampling operation
   self.module = tensorResampling
   self.type = 'gaussian'
   --self.modules = {self.module}
end

-- The temporal attention sensor will focus on a location at the x coord of the center of the output glimpse
function TemporalGlimpse:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable >= 2)
   -- Separate the input and location 
   local input, location = unpack(inputTable)
   input, location = self:toBatch(input, 1), self:toBatch(location, 0)
   --assert(input:dim() == 2 and location:dim() == 1)
   -- Create the output (successive glimpses)
   self.output:resize(input:size(1), self.depth, self.size)
   -- Cropping and padding
   self._crop = self._crop or self.output.new()
   self._pad = self._pad or input.new()
   -- For each sample in the batch
   for sampleIdx=1,self.output:size(1) do
      local outputSample = self.output[sampleIdx]
      local inputSample = input[sampleIdx]
      local pos = location[sampleIdx]
      -- (-1) far left, (1) far right of a time series, rescale to [0, 1]
      pos = (pos[1] + 1) / 2
      -- For each depth of glimpse : pad, crop, downscale
      local glimpseSize = self.size
      for depth=1,self.depth do 
         local dst = outputSample[depth]
         -- Factor to which we will crop and rescale
         if depth > 1 then glimpseSize = glimpseSize * self.scale end
         -- Add zero padding (glimpse could be partially out of bounds)
         local padSize = math.floor((glimpseSize - 1) / 2)
         -- Create a tensor of zeros (padding)
         self._pad:resize(1, input:size(2) + padSize * 2):zero()
         local center = self._pad:narrow(2, padSize + 1, input:size(2));
         center:copy(inputSample)
         -- Crop it
         local h = self._pad:size(2) - glimpseSize;
         local x = math.min(h, math.max(0, pos * h));
         -- At first depth, no downscaling
         if depth == 1 then
            dst:copy(self._pad:narrow(2, x + 1, glimpseSize));
         else
            self._crop:resize(1, glimpseSize)
            self._crop:copy(self._pad:narrow(2, x + 1, glimpseSize))
            -- Finally resample the cropped tensor
            dst:copy(tensorResampling(self._crop, self.size, 'gaussian'));
         end
      end
   end
   -- Finally resize the output
   self.output:resize(input:size(1), self.depth, self.size)
   self.output = self:fromBatch(self.output, 1)
   return self.output
end

function TemporalGlimpse:updateGradInput(inputTable, gradOutput)
   -- Separate the input and location 
   local input, location = unpack(inputTable)
   input, location = self:toBatch(input, 1), self:toBatch(location, 0)
   -- Prepare the gradient sizes to match the input 
   local gradInput, gradLocation = unpack(self.gradInput)
   gradOutput = self:toBatch(gradOutput, 1)
   gradInput:resizeAs(input):zero()
   gradLocation:resizeAs(location):zero() -- no backprop through location
   -- Prepare the gradient w.r.t the output
   gradOutput = gradOutput:view(input:size(1), self.depth, self.size)
   for sampleIdx=1,self.output:size(1) do
      local gradOutputSample = gradOutput[sampleIdx]
      local gradInputSample = gradInput[sampleIdx]
      local pos = location[sampleIdx]
      -- (-1) far left, (1) far right of a time series, rescale to [0, 1]
      pos = (pos[1] + 1) / 2
      -- For each depth of glimpse : pad, crop, downscale
      local glimpseSize = self.size
      for depth=1,self.depth do 
         local src = gradOutputSample[depth]
         -- Factor to which we will crop and rescale
         if depth > 1 then glimpseSize = glimpseSize * self.scale end
         -- Add zero padding (glimpse could be partially out of bounds)
         local padSize = math.floor((glimpseSize - 1) / 2)
         -- Create a tensor of zeros (padding)
         self._pad:resize(1, input:size(2) + padSize * 2):zero()
         -- Crop it
         local h = self._pad:size(2) - glimpseSize;
         local x = math.min(h, math.max(0, pos * h));
         local pad = self._pad:narrow(2, x + 1, glimpseSize)
         -- At first depth, no downscaling
         if depth == 1 then
            pad:copy(src);
         else
            self._crop:resize(1, glimpseSize)
            -- Finally copy the derivative of the resampling ! NOT DONE
            -- ad:copy(gradInput(tensorResampling(self._crop, self.size, 'gaussian')));
         end
         gradInputSample:add(self._pad:narrow(2, padSize+1, input:size(2)))
      end
   end
   -- Finally set the gradients
   self.gradInput[1] = self:fromBatch(gradInput, 1)
   self.gradInput[2] = self:fromBatch(gradLocation, 0)
   return self.gradInput
end

local modelRAM, parent = torch.class('modelRAM', 'modelRNN')

function modelRAM:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  --[[ Glimpse network (rnn input layer) ]]--
  -- Location sensor 
  locationSensor = nn.Sequential()
  locationSensor:add(nn.SelectTable(2))
  locationSensor:add(nn.Linear(1, self.locatorHiddenSize))
  locationSensor:add(self.nonLinearity())
  -- Glimpse sensor
  glimpseSensor = nn.Sequential()
  glimpseSensor:add(nn.TemporalGlimpse(self.glimpseSize, self.glimpseDepth, self.glimpseScale))
  --glimpseSensor:add(nn.Collapse(3))
  glimpseSensor:add(nn.Reshape(self.glimpseSize * self.glimpseDepth));
  glimpseSensor:add(nn.Linear(self.glimpseSize * self.glimpseDepth, self.glimpseHiddenSize))
  glimpseSensor:add(self.nonLinearity())
  -- Complete glimpse network
  glimpse = nn.Sequential();
  glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor));
  glimpse:add(nn.JoinTable(1,1));
  glimpse:add(nn.Linear(self.glimpseHiddenSize + self.locatorHiddenSize, self.imageHiddenSize));
  glimpse:add(self.nonLinearity());
  glimpse:add(nn.Linear(self.imageHiddenSize, self.hiddenSize));
  -- Rnn recurrent layer
  recurrent = nn.Linear(self.hiddenSize, self.hiddenSize);
  -- Recurrent neural network
  rnn = nn.Recurrent(self.hiddenSize, glimpse, recurrent, self.nonLinearity(), 99999)
  seriesSize = structure.nInputs
  -- actions (locator)
  locator = nn.Sequential()
  locator:add(nn.Linear(self.hiddenSize, 1))
  locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
  locator:add(nn.ReinforceNormal(2*self.locatorStd, self.stochastic)) -- sample from normal, uses REINFORCE learning rule
  assert(locator:get(3).stochastic == self.stochastic, "Please update the dpnn package : luarocks install dpnn")
  locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
  --locator:add(nn.MulConstant(self.unitPixels*2 / seriesSize))
  -- Final recurrent attention model
  attention = nn.RecurrentAttention(rnn, locator, self.rho, {self.hiddenSize})
  -- model is a reinforcement learning agent
  agent = nn.Sequential()
  -- agent:add(nn.Convert(ds:ioShapes(), 'bchw'))
  agent:add(attention)
  -- classifier :
  agent:add(nn.SelectTable(-1))
  agent:add(nn.Linear(self.hiddenSize, structure.nOutputs))
  agent:add(nn.LogSoftMax())
  -- add the baseline reward predictor
  seq = nn.Sequential()
  seq:add(nn.Constant(1,1))
  seq:add(nn.Add(1))
  concat = nn.ConcatTable():add(nn.Identity()):add(seq)
  concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)
  -- output will be : {classpred, {classpred, basereward}}
  agent:add(concat2)
  return agent;
end

function modelRAM:definePretraining(structure, l, options)
  local model = {};
  -- Return the complete model
  return model;
end

function modelRAM:retrieveEncodingLayer(model)
  -- Retrieve only the encoding layer 
  local encoder = model.encoder
  return encoder
end

function modelRAM:defineCriterion(model)
  local loss = nn.ParallelCriterion(true);
  loss:add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
  loss:add(nn.ModuleCriterion(nn.VRClassReward(model, self.rewardScale), nil, nn.Convert())) -- REINFORCE
  return model, loss;
end
  
function modelRAM:weightsInitialize(model)
  -- Initialize the batch normalization layers
  for k,v in pairs(model:findModules('nn.BatchNormalization')) do
    v.weight:fill(1)
    v.bias:zero()
  end
  -- Find only the linear modules (including LSTM's)
  --linearNodes = model:findModules('nn.Linear')
  --for l = 1,#linearNodes do
  --  module = linearNodes[l];
  --  module:init('weight', self.initialize);
  --  module:init('bias', self.initialize);
  --end
  return model;
end

function modelRAM:weightsTransfer(model, trainedLayers)
  -- Find both LSTM and linear modules
  linearNodes = model:findModules('nn.Linear');
  -- Current linear layer
  local curLayer = 1;
  for l = 1,#trainedLayers do
    -- Find equivalent in pre-trained layer
    lstmNodes = trainedLayers[l].encoder:findModules('nn.Linear');
    for k = 1,#lstmNodes do
      linearNodes[curLayer].weights = lstmNodes[k].weight;
      linearNodes[curLayer].bias = lstmNodes[k].bias;
      curLayer = curLayer + 1;
    end
  end
  return model;
end

function modelRAM:parametersDefault()
  self.initialize = nninit.xavier;
  self.addNonLinearity = true;
  self.layerwiseLinear = true;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  --[[ reinforce ]]--
  self.rewardScale = 1;           -- scale of positive reward (negative is 0)
  self.unitPixels = 62;            -- locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)
  self.locatorStd = 0.11;         -- stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)
  self.stochastic = false;        -- reinforce modules forward inputs stochastically during evaluation
  --[[ glimpse layer ]]--
  self.glimpseHiddenSize = 128    -- size of glimpse hidden layer
  self.glimpseSize = 32           -- size of glimpse at highest granularity
  self.glimpseScale = 1.5         -- scale of successive glimpses w.r.t. original dimensionality
  self.glimpseDepth = 4           -- number of concatenated downscaled patches
  self.locatorHiddenSize = 128    -- size of locator hidden layer
  self.imageHiddenSize = 256      -- size of hidden layer combining glimpse and locator hiddens
  --[[ recurrent layer ]]--
  self.rho = 7                    -- back-propagate through time (BPTT) for rho time-steps
  self.hiddenSize = 256           -- number of hidden units used in Simple RNN
  self.pretrain = false;
  self.windowSize = 16;
  self.windowStep = 1;
  self.dropout = 0.5;
end

function modelRAM:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end