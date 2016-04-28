------------------------------------------------------------------
-- istaencoder class (possibly with regularizer on code)


----------------------------------------------------------------------
-- Imports
require 'nn'
require 'torch'
require 'modelClass'
local nninit = require 'nninit'

------------------------------------------------------------------
local istastep, parent = torch.class('nn.istastep', 'nn.Module')

function istastep:__init(outputsize)
    parent.__init(self)
    self.S = nn.Linear(outputsize, outputsize)
    self.G = nn.SoftShrink()
    self.hidden = nil
    self.gradCode = nil
    self.gradZ = nil
end

function istastep:updateOutput(z, code)
    self.hidden = torch.add(code, self.S:updateOutput(z))
    self.output = self.G:updateOutput(self.hidden)
    return self.output
end

function istastep:updateGradInput(z, code, gradOutput)
    self.gradCode = self.G:updateGradInput(self.hidden, gradOutput)
    self.gradZ = self.S:updateGradInput(z, self.gradCode)
    self.gradInput = self.gradZ, self.gradCode
   return self.gradZ, self.gradCode
end

function istastep:accGradParameters(z, code, gradOutput)
    self.G:accGradParameters(self.hidden, gradOutput)
    self.S:accGradParameters(z, self.gradCode)
end

function istastep:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i = 1, #from do tinsert(to, from[i]) end
        else
            table.insert(to, from)
        end
    end
    local w = {}
    local gw = {}
    local mw, mgw = self.S:parameters()
    if mw then tinsert(w, mw) tinsert(gw, mgw) end
    local mw, mgw = self.G:parameters()
    if mw then tinsert(w, mw) tinsert(gw, mgw) end
    return w, gw
end

------------------------------------------------------------------
local istaencoder, parent = torch.class('nn.istaencoder', 'nn.Module')

function istaencoder:__init(inputsize, outputsize, n)
    parent.__init(self)
    self.encoder = nn.Linear(inputsize, outputsize)
    self.G0 = nn.SoftShrink()
    -- array of ista step modules
    self.istasteps = {}
    self.code0 = nil
    self.gradCode0 = nil
    for i = 1, n do
        self.istasteps[i] = nn.istastep(outputsize)
    end
end

function istaencoder:updateOutput(input)
    -- print(input:size())
    -- print('updating output')
    self.code0 = self.encoder:updateOutput(input)
    local z = self.G0:updateOutput(self.code0)
    for i = 1, #self.istasteps do
        z = self.istasteps[i]:updateOutput(z, self.code0)
    end
    self.output = z
    return self.output
end

function istaencoder:updateGradInput(input, gradOutput)
    -- print('updating grad input')
    local gradZ = gradOutput
    local gradTemp = nil
    if self.gradCode0 == nil then self.gradCode0 = torch.Tensor(gradOutput:size()) end
    self.gradCode0:zero()
    if #self.istasteps > 1 then
        for i = #self.istasteps, 2, -1 do
            gradZ, gradTemp = self.istasteps[i]:updateGradInput(self.istasteps[i - 1].output, self.code0, gradZ)
            self.gradCode0 = torch.add(self.gradCode0, gradTemp)
        end
        gradZ, gradTemp = self.istasteps[1]:updateGradInput(self.G0.output, self.code0, gradZ)
        self.gradCode0 = torch.add(self.gradCode0, gradTemp)
    end
    gradTemp = self.G0:updateGradInput(self.code0, gradZ)
    self.gradCode0 = torch.add(self.gradCode0, gradTemp)
    self.gradInput = self.encoder:updateGradInput(input, self.gradCode0)
    return self.gradInput
end

function istaencoder:accGradParameters(input, gradOutput)
    local gradZ = gradOutput
    local gradTemp = nil
    if #self.istasteps > 1 then
        for i = #self.istasteps, 2, -1 do
            self.istasteps[i]:accGradParameters(self.istasteps[i - 1].output, self.code0, gradZ)
            gradZ = self.istasteps[i].gradInput
        end
        self.istasteps[1]:accGradParameters(self.G0.output, self.code0, gradZ)
        gradZ = self.istasteps[1].gradInput
    end
    self.G0:accGradParameters(self.code0, gradZ)
    self.encoder:accGradParameters(input, self.gradCode0)
end

-- collect the parameters so they can be flattened
-- this assumes that the cost doesn't have parameters.
function istaencoder:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i = 1, #from do tinsert(to, from[i]) end
        else
            table.insert(to, from)
        end
    end
    local w = {}
    local gw = {}
    local mw, mgw = self.encoder:parameters()
    if mw then tinsert(w, mw) tinsert(gw, mgw) end
    local mw, mgw = self.G0:parameters()
    if mw then tinsert(w, mw) tinsert(gw, mgw) end
    for i = 1, #self.istasteps do
        local mw, mgw = self.istasteps[i]:parameters()
        if mw then tinsert(w, mw) tinsert(gw, mgw) end
    end
    return w, gw
end


function istaencoder:weights()
    return self.encoder.weight
end

------------------------------------------------------------------
-- an auto-encoder with a regularizer on the code vector
------------------------------------------------------------------
local registaencoder, parent = 
    torch.class('nn.registaencoder', 'nn.istaencoder')

function registaencoder:__init(encoder, decoder, cost, regularizer)
    parent.__init(self)
    self.encoder = encoder
    self.decoder = decoder
    self.cost = cost
    self.regularizer = regularizer   -- regularizer module
    self.code = 0
    self.gradcode = 0
    self.recons = 0
    self.gradrecons = 0
    self.alpha = 1.0         -- coefficient of regularizer
    self.recenergy = 0       -- reconstruction energy
    self.regenergy = 0       -- regularizer energy
    self.energy = 0          -- total energy (sum of the above)
end

function registaencoder:updateOutput(input)
    self.code = self.encoder:updateOutput(input)
    self.regenergy = self.regularizer:updateOutput(self.code)
    self.recons = self.decoder:updateOutput(self.code)
    self.recenergy = self.cost:updateOutput(self.recons, input)
    self.energy = self.regenergy + self.recenergy
    return self.energy
end

function registaencoder:updateGradInput(input)
    self.gradrecons = self.cost:updateGradInput(self.recons, input)
    self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons) +
                    self.regularizer:updateGradInput(self.code)
    return self.encoder:updateGradInput(input, self.gradcode)
end

function registaencoder:accGradParameters(input)
    self.gradrecons = self.cost:updateGradInput(self.recons, input)
    -- self.gradrecons = self.cost:accGradParameters(recons, input)
    self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons) +
                    self.regularizer:updateGradInput(self.code)
    self.decoder:accGradParameters(self.code, self.gradrecons)
    self.encoder:updateGradInput(input, self.gradcode)
    self.encoder:accGradParameters(input, self.gradcode)
end


function nn.Linear:normalize()
    for i=1,self.weight:size(2) do
        local col = self.weight:select(2, i)
        local norm = torch.norm(col)
        if norm > 1 then torch.mul(col, col, 1/norm) end
    end
end

local autoencoder, parent = torch.class('nn.autoencoder', 'nn.Module')

function autoencoder:__init(encoder, decoder, cost)
   parent.__init(self)
   self.encoder = encoder   -- encoder module
   self.decoder = decoder   -- decoder module
   self.cost = cost         -- reconstruction cost module
   self.code = 0            -- code vector
   self.gradcode = 0        -- gradient w.r.t. code
   self.recons = 0          -- reconstruction
   self.gradrecons = 0      -- gradient w.r.t. reconstruction
   self.energy = 0          -- reconstruction energy
end

function autoencoder:updateOutput(input)
   self.code = self.encoder:updateOutput(input)
   self.recons = self.decoder:updateOutput(self.code)
   self.energy = self.cost:updateOutput(self.recons, input)
   return self.energy
end

function autoencoder:updateGradInput(input)
   self.gradrecons = self.cost:updateGradInput(self.recons, input)
   self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons)
   return self.encoder:updateGradInput(input, self.gradcode)
end

function autoencoder:accGradParameters(input)
   self.gradrecons = self.cost:updateGradInput(self.recons, input)
   -- self.gradrecons = self.cost:accGradParameters(recons, input)
   self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons)
   self.decoder:accGradParameters(self.code, self.gradrecons)
   self.encoder:accGradParameters(input, self.gradcode)
end

function autoencoder:normalize()
    return self.decoder:normalize()
end

-- collect the parameters so they can be flattened
-- this assumes that the cost doesn't have parameters.
function autoencoder:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do tinsert(to,from[i]) end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   local mw,mgw = self.encoder:parameters()
   if mw then tinsert(w,mw) tinsert(gw,mgw) end
   local mw,mgw = self.decoder:parameters()
   if mw then tinsert(w,mw) tinsert(gw,mgw) end
   return w,gw
end


function autoencoder:weights()
  return decoder.weight, module.encoder:weights()
end

function nn.Sequential:weights()
  return module.encoder.modules[1].weight
end

-- args: number columns of filters, 
-- width of each filter, height of each filter.
-- product of width and height must equal dimension of input.
function autoencoder:displayweights(ncol,w,h)
  local dw,ew = self.weights()
  dw = dw:transpose(1,2):unfold(2,h,w)
  ew = ew:unfold(2,h,w)
  dd = image.toDisplayTensor{input=dw,
                             padding=2,
                             nrow=ncol,
                             symmetric=true}
  de = image.toDisplayTensor{input=ew,
                             padding=2,
                             nrow=ncol,
                             symmetric=true}
  return dd,de
end

------------------------------------------------------------------
-- an auto-encoder with a regularizer on the code vector
local regautoencoder, parent = 
  torch.class('nn.regautoencoder', 'nn.autoencoder')

function regautoencoder:__init(encoder, decoder, cost, regularizer)
   parent.__init(self)
   self.encoder = encoder
   self.decoder = decoder
   self.cost = cost
   self.regularizer = regularizer   -- regularizer module
   self.code = 0
   self.gradcode = 0
   self.recons = 0
   self.gradrecons = 0
   self.recenergy = 0       -- reconstruction energy
   self.regenergy = 0       -- regularizer energy
   self.energy = 0          -- total energy (sum of the above)
end

function regautoencoder:updateOutput(input)
   self.code = self.encoder:updateOutput(input)
   self.regenergy = self.regularizer:updateOutput(self.code)
   self.recons = self.decoder:updateOutput(self.code)
   self.recenergy = self.cost:updateOutput(self.recons, input)
   self.energy = self.regenergy + self.recenergy
   return self.energy
end

function regautoencoder:updateGradInput(input)
   self.gradrecons = self.cost:updateGradInput(self.recons, input)
   self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons) +
                   self.regularizer:updateGradInput(self.code)
   return self.encoder:updateGradInput(input, self.gradcode)
end

function regautoencoder:accGradParameters(input)
   self.gradrecons = self.cost:updateGradInput(self.recons, input)
   -- self.gradrecons = self.cost:accGradParameters(recons, input)
   self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons) +
                   self.regularizer:updateGradInput(self.code)
   self.decoder:accGradParameters(self.code, self.gradrecons)
   self.encoder:updateGradInput(input, self.gradcode)
   self.encoder:accGradParameters(input, self.gradcode)
end

------------------------------------------------------------------
-- L1 over L2 criterion for regularization
------------------------------------------------------------------
local L1overL2Criterion, parent = torch.class('nn.L1overL2Criterion', 'nn.Criterion')

function L1overL2Criterion:__init(alpha, beta)
    parent.__init(self)
    self.alpha = alpha
    self.beta = beta or 0.1
end

-- calculates the L1 norm over the L2 norm of input and returns it
function L1overL2Criterion:updateOutput(input)
    self.output = self.alpha * (torch.norm(input, 1) + self.beta)/ math.sqrt(torch.norm(input)^2 + self.beta^2)
    return self.output
end

-- calculates the gradient of the L1 norm over the L2 norm of input w.r.t input and returns it
function L1overL2Criterion:updateGradInput(input)
    local l1 = torch.norm(input, 1) + self.beta
    local l2 = math.sqrt(torch.norm(input)^2 + self.beta^2)
    self.gradInput = torch.add(torch.mul(torch.sign(input), self.alpha / l2), torch.mul(input, -self.alpha * l1 / (l2^3)))
    return self.gradInput
end

----------------------------------------------------------------------
-- Definition of ISTA autoencoder (as MLP)
----------------------------------------------------------------------

local modelISTA, parent = torch.class('modelISTA', 'modelClass')

function modelISTA:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  --model:add(nn.Reshape(structure.nInputs));
  -- Hidden layers
  for i = 1,structure.nLayers do
    -- Linear transform
    if i == 1 then
      model:add(nn.Linear(structure.nInputs,structure.layers[i]));
    else
      model:add(nn.Linear(structure.layers[i-1],structure.layers[i]));
    end
    -- Batch normalization
    if self.batchNormalize then model:add(nn.BatchNormalization(structure.layers[i])); end
    -- Non-linearity
    model:add(self.nonLinearity());
    -- Dropout
    if self.dropout then model:add(nn.Dropout(self.dropout)); end
  end
  -- Final regression layer
  model:add(nn.Linear(structure.layers[structure.nLayers],structure.nOutputs)) 
  return model;
end

function modelISTA:definePretraining(structure, l, options)
  -- Prepare the layer properties
  if l == 1 then inS = structure.nInputs; else inS = structure.layers[l - 1]; end
  outS = structure.layers[l]; 
  --[[ Encoder part ]]--
  local encoder = nn.Sequential();
  -- Linear transform
  encoder:add(nn.Linear(inS, outS));
  -- Batch normalization
  if self.batchNormalize then encoder:add(nn.BatchNormalization(outS)); end
  -- Non-linearity
  encoder:add(self.nonLinearity());
  -- Dropout
  if self.dropout then encoder:add(nn.Dropout(self.dropout)) end
  --encoder:add(nn.Diag(outS));
  -- decoder
  local decoder = nn.Sequential();
  decoder:add(nn.Linear(outS,inS));
  -- impose weight sharing
  decoder:get(1).weight = encoder:get(1).weight:t();
  decoder:get(1).gradWeight = encoder:get(1).gradWeight:t();
  -- define the cost
  local cost = nn.MSECriterion()
  -- define the regularizer
  local regularizer = nn.L1overL2Criterion(self.alpha, self.beta)
  -- complete model
  return nn.regautoencoder(encoder, decoder, cost, regularizer)
end

function modelISTA:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder
  encoder:remove();
  return model.encoder;
end

function modelISTA:weightsInitialize(model)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', self.initialize);
    module:init('bias', self.initialize);
  end
  return model;
end

function modelISTA:weightsTransfer(model, trainedLayers)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#trainedLayers do
    -- Find equivalent in pre-trained layer
    preTrained = trainedLayers[l].encoder:findModules('nn.Linear');
    linearNodes[l].weight = preTrained[1].weight;
    linearNodes[l].bias = preTrained[1].bias;
  end
  return model;
end

function modelISTA:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.pretrain = true;
  self.dropout = 0.5;
  self.alpha = 0.1;
  self.beta = 0.1;
end

function modelISTA:parametersRandom()
  -- All possible non-linearities
  self.distributions = {};
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
  self.distributions.batchNormalize = {true, false};
  self.distributions.pretrainType = {'ae', 'psd'};
  self.distributions.pretrain = {true, false};
  self.distributions.dropout = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
end
