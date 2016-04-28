----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for residual network
--
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'image'
local nninit = require 'nninit'

local modelResidual, parent = torch.class('modelResidual', 'modelCNN')

----------------------------------------------------------------------
-- Temporal version of Spatial Transformer network
----------------------------------------------------------------------
function modelResidual:defineModel(structure, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- main 
  local depth = self.depth or 34;
  local shortcutType = self.shortcutType or 'A'
  local iChannels
  -- The shortcut layer is either identity or 1x1 convolution
  local function shortcut(nInputPlane, nOutputPlane, stride)
    local useConv = shortcutType == 'C' or (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
    if useConv then
      -- 1x1 convolution
      return nn.Sequential()
        :add(nn.TemporalConvolution(nInputPlane, nOutputPlane, 1, stride))
        :add(nn.BatchNormalization(nOutputPlane))
    elseif nInputPlane ~= nOutputPlane then
      -- Strided, zero-padded identity shortcut
      return nn.Sequential()
        :add(nn.TemporalMaxPooling(1, stride))
        :add(nn.Concat(2):add(nn.Identity()):add(nn.MulConstant(0)))
    else
      return nn.Identity()
    end
  end
  -- The basic residual layer block
  local function basicblock(n, stride, nRes)
    local nInputPlane = iChannels
    iChannels = n
    -- The convolutional path
    local s = nn.Sequential()
    s:add(nn.Padding(2, -1)); s:add(nn.Padding(2, 1));
    s:add(nn.TemporalConvolution(nInputPlane,n,3,stride))
    s:add(nn.Reshape(nRes * n)); s:add(nn.BatchNormalization(nRes * n)); s:add(nn.Reshape(nRes, n));
    s:add(self.nonLinearity())
    s:add(nn.Padding(2, -1)); s:add(nn.Padding(2, 1));
    s:add(nn.TemporalConvolution(n, n, 3, 1))
    s:add(nn.Reshape(nRes * n)); s:add(nn.BatchNormalization(nRes * n)); s:add(nn.Reshape(nRes, n));
    -- The shortcut (bypass) + add both torgether
    return nn.Sequential()
      :add(nn.ConcatTable()
        :add(s)
        :add(shortcut(nInputPlane, n, stride)))
      :add(nn.CAddTable(true))
      :add(self.nonLinearity())
  end
  -- The bottleneck residual layer block
  local function bottleneck(n, stride, nRes)
    local nInputPlane = iChannels
    iChannels = n * 4
    -- The convolutional path
    local s = nn.Sequential()
    s:add(nn.TemporalConvolution(nInputPlane,n,1,1))
    s:add(nn.Reshape(nRes * n)); s:add(nn.BatchNormalization(nRes * n)); s:add(nn.Reshape(nRes, n));
    s:add(self.nonLinearity())
    s:add(nn.Padding(2, -1)); s:add(nn.Padding(2, 1));
    s:add(nn.TemporalConvolution(n, n, 3, stride))
    s:add(nn.Reshape(nRes * n)); s:add(nn.BatchNormalization(nRes * n)); s:add(nn.Reshape(nRes, n));
    s:add(self.nonLinearity())
    s:add(nn.TemporalConvolution(n, n * 4, 1, 1))
    s:add(nn.BatchNormalization(n * 4))
    -- The shortcut (bypass) + add both torgether
    return nn.Sequential()
      :add(nn.ConcatTable()
        :add(s)
        :add(shortcut(nInputPlane, n * 4, stride)))
      :add(nn.CAddTable(true))
      :add(self.nonLinearity())
  end
  -- Creates count residual blocks with specified number of features
  local function layer(block, features, count, stride, nRes)
    local s = nn.Sequential()
    for i=1,count do
      s:add(block(features, i == 1 and stride or 1, nRes))
      s:add(nn.PReLU());
    end
    return s
  end
  -- Final full residual model
  local model = nn.Sequential();
  -- First reshape the input
  model:add(nn.Reshape(structure.nInputs, 1));
  if self.type == 'large' then
    --  num. residual blocks, num features, residual block function
    local cfg = {
      [18]  = {{2, 2, 2, 2}, 512, basicblock},
      [34]  = {{3, 4, 6, 3}, 512, basicblock},
      [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
      [101] = {{3, 4, 23, 3}, 2048, bottleneck},
      [152] = {{3, 8, 36, 3}, 2048, bottleneck},
    }
    -- Check that current depth is available
    assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
    -- Retrieve the corresponding features
    local def, nFeatures, block = table.unpack(cfg[depth])
    -- Number of channels
    iChannels = 64
    print(' | Residual large ' .. depth .. ' layers')
    -- The ResNet ImageNet model
    model:add(nn.Padding(2, -2)); model:add(nn.Padding(2, 3));
    model:add(nn.TemporalConvolution(1, 64, 7, 2, 3));
    model:add(nn.Reshape(64 * 64));
    model:add(nn.BatchNormalization(64 * 64));
    model:add(nn.Reshape(64, 64));
    model:add(self.nonLinearity());
    model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1));
    model:add(nn.TemporalMaxPooling(3, 1));
    model:add(layer(block, 64, def[1], 1, structure.nInputs / 2))
    model:add(layer(block, 128, def[2], 2, structure.nInputs / 4))
    model:add(layer(block, 256, def[3], 2, structure.nInputs / 8))
    model:add(layer(block, 512, def[4], 2, structure.nInputs / 16))
    model:add(nn.TemporalMaxPooling(7, 1))
    model:add(nn.Reshape(nFeatures * (structure.nInputs / 64)))
    model:add(nn.Linear(nFeatures * (structure.nInputs / 64), structure.nOutputs))
  elseif self.type == 'small' then
    -- Model type specifies number of layers for CIFAR-10 model
    assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
    local n = (depth - 2) / 6
    iChannels = 16
    print(' | Residual small ' .. depth .. ' layers')
    -- The ResNet CIFAR-10 model
    model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1));
    model:add(nn.TemporalConvolution(1, 16, 3, 1, 1))
    model:add(nn.Reshape(128 * 16));
    model:add(nn.BatchNormalization(128 * 16))
    model:add(nn.Reshape(128, 16));
    model:add(self.nonLinearity())
    model:add(layer(basicblock, 16, n, 1, structure.nInputs))
    model:add(layer(basicblock, 32, n, 2, structure.nInputs / 2))
    model:add(layer(basicblock, 64, n, 2, structure.nInputs / 4))
    model:add(nn.TemporalMaxPooling(8, 8, 1, 1))
    model:add(nn.Reshape(64 * (structure.nInputs / 32)))
    model:add(nn.Linear(64 * (structure.nInputs / 32), structure.nOutputs))
  else
    error('invalid residual network type : ' .. self.type)
  end
  -- Function to init weights of convolution
  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.outputFrameSize
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- Function to init weights of batch normalization
  local function BNInit(name)
    for k,v in pairs(model:findModules(name)) do
      v.weight:fill(1)
      v.bias:zero()
    end
  end
  -- Perform weight initialization
  ConvInit('cunn.TemporalConvolution')
  ConvInit('nn.TemporalConvolution')
  BNInit('fbnn.BatchNormalization')
  BNInit('cunn.BatchNormalization')
  BNInit('nn.BatchNormalization')
  -- Perform linear layer initialization
  for k,v in pairs(model:findModules('nn.Linear')) do
    v.bias:zero()
  end
  if options.cuda and options.cunn == 'deterministic' then
    model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
  end
  --model:get(1).gradInput = nil
  return model
end

function modelResidual:definePretraining(structure, l, options)
  -- TODO
  return model;
end

function modelResidual:parametersDefault()
  self.type = 'small'
  self.depth = 20;
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.RReLU;
  self.batchNormalize = true;
  self.kernelWidth = {};
  self.pretrain = false;
  self.padding = true;
  self.dropout = 0.5;
end

function modelResidual:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end