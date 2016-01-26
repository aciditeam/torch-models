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
require 'nninit'
require 'modelClass'

----------------------------------------------------------------------
-- Handmade LSTM
-- (From the Oxford course)
----------------------------------------------------------------------
function defineLSTMLayer(input_size, rnn_size, params)
  local n = params.n or 1
  local dropout = params.dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

local modelLSTM, parent = torch.class('modelLSTM', 'modelClass')

function modelLSTM:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  model:add(nn.Reshape(structure.nInputs));
  -- Hidden layers
  for i = 1,structure.nLayers do
    -- Linear transform
    if i == 1 then
      model:add(nn.FastLSTM(structure.nInputs,structure.layers[i]));
    else
      model:add(nn.FastLSTM(structure.layers[i-1],structure.layers[i]));
    end
    model:add(nn.Linear(structure.layers[i], structure.layers[i]))
    -- Batch normalization
    --if self.batchNormalize then model:add(nn.BatchNormalization(structure.layers[i])); end
    -- Non-linearity
    --if self.addNonLinearity then model:add(self.nonLinearity()); end
    -- Dropout
    --if self.dropout then model:add(nn.Dropout(self.dropout)); end
  end
  -- Final regression layer
  model:add(nn.Linear(structure.layers[structure.nLayers],structure.nOutputs)) 
  return nn.Sequencer(model);
end

function modelLSTM:definePretraining(inS, outS, options)
  --[[ Encoder part ]]--
end

function modelLSTM:retrieveEncodingLayer(model) 
  --
end

function modelLSTM:weightsInitialize(model)
  -- Find only the LSTM modules
  linearNodes = model:findModules('nn.FastLSTM')
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', self.initialize);
    module:init('bias', self.initialize);
  end
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', self.initialize);
    module:init('bias', self.initialize);
  end
  return model;
end

function modelLSTM:weightsTransfer(model, trainedLayers)
  -- Find both LSTM and linear modules
  lstmNodes = model:findModules('nn.FastLSTM');
  linearNodes = model:findModules('nn.Linear');
  for l = 1,trainedLayers do
    -- Find equivalent in pre-trained layer
    preLSTM = trainedLayers[l]:findModules('nn.FastLSTM');
    lstmNodes[l].weights = preLSTM[1].weight;
    lstmNodes[l].bias = preLSTM[1].bias;
    preTrained = trainedLayers[l]:findModules('nn.Linear');
    linearNodes[l].weight = preTrained[1].weight;
    linearNodes[l].bias = preTrained[1].bias;
  end
  return model;
end

function modelLSTM:parametersDefault()
  self.initialize = nn.kaiming;
  self.addNonLinearity = false;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = false;
  self.pretrain = false;
  self.dropout = 0.5;
end

function modelLSTM:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end

--[[

Other solutions are :
rnn.LSTM
rnn.FastLSTM 

rnn.LSTM(inputSize, outputSize, [rho])

Creating multi-layer LSTM network :
lstm = nn.Sequencer(
   nn.Sequential()
      :add(nn.LSTM(100,100))
      :add(nn.Linear(100,100))
      :add(nn.LSTM(100,100))
   )

]]--

--[[

RNN's with LSTM inside ^^
(Here to learn to parse computer programs !)

--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
local ptb = require('data')

-- Train 1 day and gives 82 perplexity.
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
               ] ] --

-- Trains 1h and gives test 115 perplexity.
local params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}

local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

local function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

local function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

local function main()
  g_init_gpu(arg)
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
  state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local words_per_step = params.seq_length * params.batch_size
  local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  local perps
  while epoch < params.max_max_epoch do
    local perp = fp(state_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end
  run_test()
  print("Training is over.")
end

main()



--
-- OTHER LSTM CLASSIFICATION
--
require 'rnn'
require 'optim'

batchSize = 50
rho = 5
hiddenSize = 64
nIndex = 10000

-- define the model
model = nn.Sequential()
model:add(nn.Sequencer(nn.LookupTable(nIndex, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
model:add(nn.Sequencer(nn.LogSoftMax()))
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- create a Dummy Dataset, dummy dataset (task predict the next item)
dataset = torch.randperm(nIndex)

-- offset is a convenient pointer to iterate over the dataset
offsets = {}
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)


-- method to compute a batch
function nextBatch()
  local inputs, targets = {}, {}
   for step = 1, rho do
      --get a batch of inputs
      table.insert(inputs, dataset:index(1, offsets))
      -- shift of one batch indexes
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > nIndex then
            offsets[j] = 1
         end
      end
      -- fill the batch of targets
      table.insert(targets, dataset:index(1, offsets))
   end
  return inputs, targets
end

-- get weights and loss wrt weights from the model
x, dl_dx = model:getParameters()

-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. weigths is the vector of trainable weights,
-- it extracts a mini_batch via the nextBatch method
feval = function(x_new)
  -- copy the weight if are changed
  if x ~= x_new then
    x:copy(x_new)
  end

  -- select a training batch
  local inputs, targets = nextBatch()

  -- reset gradients (gradients are always accumulated, to accommodate
  -- batch methods)
  dl_dx:zero()

  -- evaluate the loss function and its derivative wrt x, given mini batch
  local prediction = model:forward(inputs)
  local loss_x = criterion:forward(prediction, targets)
  model:backward(inputs, criterion:backward(prediction, targets))

  return loss_x, dl_dx
end

sgd_params = {
   learningRate = 0.1,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

-- cycle on data
for i = 1,1e4 do
  -- train a mini_batch of batchSize in parallel
  _, fs = optim.sgd(feval,x, sgd_params)

  if sgd_params.evalCounter % 100 == 0 then
    print('error for iteration ' .. sgd_params.evalCounter  .. ' is ' .. fs[1] / rho)
    -- print(sgd_params)
  end
end


--
-- OTHER CLASS-SEQUENCER LSTM SOMETHING
--

require 'rnn'

batchSize = 10
rho = 5
hiddenSize = 64
nIndex = 10000


function gradientUpgrade(model, x, y, criterion, learningRate, i)
  local prediction = model:forward(x)
  local err = criterion:forward(prediction, y)
   if i % 100 == 0 then
      print('error for iteration ' .. i  .. ' is ' .. err/rho)
   end
  local gradOutputs = criterion:backward(prediction, y)
  model:backward(x, gradOutputs)
  model:updateParameters(learningRate)
   model:zeroGradParameters()
end

-- Model
model = nn.Sequential()
model:add(nn.Sequencer(nn.LookupTable(nIndex, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
model:add(nn.Sequencer(nn.LogSoftMax()))

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())


-- dummy dataset (task predict the next item)
dataset = torch.randperm(nIndex)
-- this dataset represent a random permutation of a sequence between 1 and nIndex

-- define the index of the batch elements
offsets = {}
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)

lr = 0.1
for i = 1, 10e4 do
   local inputs, targets = {}, {}
   for step = 1, rho do
      --get a batch of inputs
      table.insert(inputs, dataset:index(1, offsets))
      -- shift of one batch indexes
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > nIndex then
            offsets[j] = 1
         end
      end
      -- a batch of targets
      table.insert(targets, dataset:index(1, offsets))
   end

   gradientUpgrade(model, inputs, targets, criterion, lr, i)
end


Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.

require 'nn'
require 'rnn'

torch.manualSeed(123)

version = 1.1 --supports both online and mini-batch training

-- Forward coupling: Copy encoder cell and output to decoder LSTM
function forwardConnect(encLSTM, decLSTM)
  decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[opt.inputSeqLen])
  decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[opt.inputSeqLen])
end

-- Backward coupling: Copy decoder gradients to encoder LSTM
function backwardConnect(encLSTM, decLSTM)
  encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
  encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end

function main()
  opt = {}
  opt.learningRate = 0.1
  opt.hiddenSz = 2
  opt.vocabSz = 5
  opt.inputSeqLen = 3 -- length of the encoded sequence

  -- Some example data
  local encInSeq, decInSeq, decOutSeq = torch.Tensor({{1,2,3},{3,2,1}}), torch.Tensor({{1,2,3,4},{4,3,2,1}}), torch.Tensor({{2,3,4,1},{1,2,4,3}})
  decOutSeq = nn.SplitTable(1, 1):forward(decOutSeq)
  
  -- Encoder
  local enc = nn.Sequential()
  enc:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
  enc:add(nn.SplitTable(1, 2)) --works for both online and mini-batch mode
  local encLSTM = nn.LSTM(opt.hiddenSz, opt.hiddenSz)
  enc:add(nn.Sequencer(encLSTM))
  enc:add(nn.SelectTable(-1))

  -- Decoder
  local dec = nn.Sequential()
  dec:add(nn.LookupTable(opt.vocabSz, opt.hiddenSz))
  dec:add(nn.SplitTable(1, 2)) --works for both online and mini-batch mode
  local decLSTM = nn.LSTM(opt.hiddenSz, opt.hiddenSz)
  dec:add(nn.Sequencer(decLSTM))
  dec:add(nn.Sequencer(nn.Linear(opt.hiddenSz, opt.vocabSz)))
  dec:add(nn.Sequencer(nn.LogSoftMax()))

  local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

  local encParams, encGradParams = enc:getParameters()
  local decParams, decGradParams = dec:getParameters()

  enc:zeroGradParameters()
  dec:zeroGradParameters()

  -- Forward pass
  local encOut = enc:forward(encInSeq)
  forwardConnect(encLSTM, decLSTM)
  local decOut = dec:forward(decInSeq)
  local Edec = criterion:forward(decOut, decOutSeq)

  -- Backward pass
  local gEdec = criterion:backward(decOut, decOutSeq)
  dec:backward(decInSeq, gEdec)
  backwardConnect(encLSTM, decLSTM)
  local zeroTensor = torch.Tensor(2):zero()
  enc:backward(encInSeq, zeroTensor)

  --
  -- You would normally do something like this now:
  --   dec:updateParameters(opt.learningRate)
  --   enc:updateParameters(opt.learningRate)
  --
  -- Here, we do a numerical gradient check to make sure the coupling is correct:
  --
  local tester = torch.Tester()
  local tests = {}
  local eps = 1e-5

  function tests.gradientCheck()
    local decGP_est, encGP_est = torch.DoubleTensor(decGradParams:size()), torch.DoubleTensor(encGradParams:size())

    -- Easy function to do forward pass over coupled network and get error
    function forwardPass()
      local encOut = enc:forward(encInSeq)
      forwardConnect(encLSTM, decLSTM)
      local decOut = dec:forward(decInSeq)
      local E = criterion:forward(decOut, decOutSeq)
      return E
    end

    -- Check encoder
    for i = 1, encGradParams:size(1) do
      -- Forward with \theta+eps
      encParams[i] = encParams[i] + eps
      local C1 = forwardPass()
      -- Forward with \theta-eps
      encParams[i] = encParams[i] - 2 * eps
      local C2 = forwardPass()

      encParams[i] = encParams[i] + eps
      encGP_est[i] = (C1 - C2) / (2 * eps)
    end
    tester:assertTensorEq(encGradParams, encGP_est, eps, "Numerical gradient check for encoder failed")

    -- Check decoder
    for i = 1, decGradParams:size(1) do
      -- Forward with \theta+eps
      decParams[i] = decParams[i] + eps
      local C1 = forwardPass()
      -- Forward with \theta-eps
      decParams[i] = decParams[i] - 2 * eps
      local C2 = forwardPass()

      decParams[i] = decParams[i] + eps
      decGP_est[i] = (C1 - C2) / (2 * eps)
    end
    tester:assertTensorEq(decGradParams, decGP_est, eps, "Numerical gradient check for decoder failed")
  end

  tester:add(tests)
  tester:run()
end

main()

]]--