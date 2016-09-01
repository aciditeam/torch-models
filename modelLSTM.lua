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
-- require 'moduleSlidingWindow'
require 'modelClass'

require 'modulePrinter' -- For debug
local nninit = require 'nninit'

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
      -- c,h from previous timesteps
      local prev_h = inputs[L*2+1]
      local prev_c = inputs[L*2]
      -- the input to this layer
      if L == 1 then 
	 x = OneHot(input_size)(inputs[1])
	 input_size_L = input_size
      else
	 x = outputs[(L-1)*2]
	 if dropout > 0 then  -- apply dropout, if any
	    x = nn.Dropout(dropout)(x) end 
	 input_size_L = rnn_size
      end
      -- evaluate the input sums at once for efficiency
      local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):
	 annotate{name='i2h_'..L}
      local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):
	 annotate{name='h2h_'..L}
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
      local next_c = nn.CAddTable()({
	    nn.CMulTable()({forget_gate, prev_c}),
	    nn.CMulTable()({in_gate, in_transform})
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
   -- Return final module
   return nn.gModule(inputs, outputs)
end

local modelLSTM, parent = torch.class('modelLSTM', 'modelClass')

function modelLSTM:__init(options)
   self:parametersDefault()
   self:loadParameters(options)
   self:parametersRandom()
end

function modelLSTM:defineModel(structure, options)
   -- local verbose = true  -- set to true for debug
   local function addPrint(model, ...)
      if verbose then model:add(nn.Printer(...)) end
   end
   
   -- Container
   local model = nn.Sequential()
   
   local batchMode = true
   
   local lstmInputs = structure.nInputs * structure.nFeats
   local lstmOutputs = structure.nOutputs * structure.nFeats
   if self.sequencer then
      lstmInputs = self.windowSize * structure.nFeats
      lstmOutputs = self.windowSize * structure.nFeats
   end
   
   -- Hidden layers
   for i = 1, structure.nLayers do
      -- Long Short-Term Memories
      if i == 1 then
	 -- Reshape to minibatch with a single feature
	 addPrint(model, 'Inside LSTM', 'size')
	 model:add(nn.View(-1, lstmInputs))
	 addPrint(model, 'After LSTM Input Reshape', 'size')
	 
	 curLSTM = nn.FastLSTM(lstmInputs, structure.layers[i], self.rho);
      else
	 curLSTM = nn.FastLSTM(structure.layers[i-1], structure.layers[i], self.rho)
      end
      
      -- Always initialize the bias of the LSTM forget gate to 1
      -- (trick from old RNN study)
      if self.initForget then
	 curLSTM.i2g.bias[{{2*structure.layers[i]+1,3*structure.layers[i]}}]:
	    fill(1)
      end
      
      -- Add the bias-adjusted LSTM to the network
      model:add(curLSTM)
      
      -- Layer-wise linear transform
      if self.layerwiseLinear then
	 model:add(nn.Linear(structure.layers[i],
			     structure.layers[i]))
      end
            
      -- Batch normalization
      if self.batchNormalize then
	 model:add(nn.BatchNormalization(structure.layers[i]))
      end
      -- Non-linearity
      if self.addNonLinearity then model:add(self.nonLinearity()) end
      -- Dropout
      if self.dropout then model:add(nn.Dropout(self.dropout)) end
   end
   
   -- Final regression layer for classification
   if self.sequencer then
      -- Sequencer case simply needs to add a linear transform
      -- to number of classes
      model:add(nn.Linear(structure.layers[structure.nLayers],
			  lstmOutputs))
      lstmModel = nn.Sequencer(model)
      model = nn.Sequential()
      -- Number of windows we will consider
      local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
      -- Here we add the subsequencing trick
      local tensOut = false
      addPrint(model, 'Input to sliding window', 'size')
      model:add(nn.SlidingWindow(1, self.windowSize, self.windowStep,
				 structure.nFeats, tensOut, options.cuda))
      addPrint(model, 'Sliding Window Output', true)
      
      -- if structure.nFeats > 1 then
      -- 	 model:add(nn.Reshape(nWins, 1, self.windowSize * structure.nFeats))
      -- end
      addPrint(model, 'Input to LSTM', 'size')
      model:add(lstmModel)

      addPrint(model, 'Before JoinTable', 'size')
      model:add(nn.JoinTable(2, 2))
      
      -- Reshape for final fully connected layer
      addPrint(model, 'Before reshape for final fully connected layer', 'size')
      
      model:add(nn.View(-1, nWins * self.windowSize * structure.nFeats))
      
      local outputDuration = structure.nOutputs  -- Output has duration of input
      if options.predict then
	 -- Restrict output duration to prediction
	 outputDuration = options.predictionLength
      end
      
      addPrint(model, 'Input to final fully connected layer', 'size')
      model:add(nn.Linear(nWins * self.windowSize * structure.nFeats,
			  structure.nOutputs * structure.nFeats))
      -- Reshape to format seqDuration x featsNum
      addPrint(model, 'Input to reshape to separate time and feature dimensions', 'size')
      model:add(nn.View(-1, structure.nOutputs, structure.nFeats))
      addPrint(model, 'Input to reshape to rnn', 'size')
      model:add(nn.Transpose({1, 2}))  -- Bring back to rnn convention
   else
      -- Recursor case
      lstmLayers = nn.Recursor(model)
      model = nn.Sequential()
      -- Add the LSTM layers
      model:add(lstmLayers)
      -- Needs to reshape the data from all outputs
      model:add(nn.View(structure.layers[structure.nLayers]))
      -- And then add linear transform to number of classes
      model:add(nn.Linear(structure.layers[structure.nLayers],
			  structure.nOutputs))
   end
   addPrint(model, 'Output', 'size')
   return model
end

function modelLSTM:definePretraining(structure, l, options)
   --[[ Encoder part ]]--
   local finalEncoder = nn.Sequential()
   local encoder = nn.Sequential()
   if l == 1 then
      if (self.sequencer) then
	 nIn = self.windowSize
      else
	 nIn = structure.nInputs
      end
   else
      nIn = structure.layers[l-1]
   end
   curLSTM = nn.FastLSTM(nIn, structure.layers[l], self.rho)
   -- Always initialize the bias of the LSTM forget gate to 1
   -- (trick from old RNN study)
   if self.initForget then
      curLSTM.i2g.bias[{{2*structure.layers[l]+1,
			 3*structure.layers[l]}}]:fill(1)
   end
   -- Add the bias-adjusted LSTM to the network
   encoder:add(curLSTM)
   -- Layer-wise linear transform
   if self.layerwiseLinear then
      encoder:add(nn.Linear(structure.layers[l],
			    structure.layers[l]))
   end
   -- Batch normalization
   if self.batchNormalize then
      encoder:add(nn.BatchNormalization(structure.layers[l]))
   end
   -- Non-linearity
   if self.addNonLinearity then encoder:add(self.nonLinearity()) end
   -- Dropout
   if self.dropout then encoder:add(nn.Dropout(self.dropout)) end
   -- Perform recurrent encoder
   encoder = nn.Sequencer(encoder)
   -- In the first layer we have to perform a Sliding Window
   if l == 1 then
      -- Number of windows we will consider
      local nWins = torch.ceil(
	 (structure.nInputs - self.windowSize + 1) / self.windowStep)
      -- Here we add the subsequencing trick
      finalEncoder:add(nn.SlidingWindow(2, self.windowSize, self.windowStep))
   end
   finalEncoder:add(encoder)
   --[[ Decoder part ]]--
   local decoder = nn.Sequential()
   local finalDecoder = nn.Sequential()
   local decLSTM = nn.FastLSTM(structure.layers[l], structure.layers[l])
   decoder:add(nn.Sequencer(decLSTM))
   decoder:add(nn.Sequencer(nn.Linear(structure.layers[l], nIn)))
   finalDecoder:add(decoder)
   -- In the first layer we have to join windows
   if l == 1 then
      -- Number of windows we will consider
      local nWins = torch.ceil(
	 (structure.nInputs - self.windowSize + 1) / self.windowStep)
      -- Here we add the subsequencing trick
      finalDecoder:add(nn.JoinTable(2))
      finalDecoder:add(nn.Linear(nWins * self.windowSize, structure.nInputs))
   end
   -- Construct an autoencoder
   local model = unsup.AutoEncoder(finalEncoder, finalDecoder, options.beta)
   -- We will need a sequencer criterion for deeper layers
   if (l > 1) then model.loss = nn.SequencerCriterion(model.loss) end
   -- Return the complete model
   return model
end

function modelLSTM:retrieveEncodingLayer(model)
   -- Retrieve only the encoding layer 
   encoder = model.encoder
   return encoder
end

function modelLSTM:weightsInitialize(model)
   -- Find only the linear modules (including LSTM's)
   linearNodes = model:findModules('nn.Linear')
   for l = 1,#linearNodes do
      module = linearNodes[l]
      module:init('weight', self.initialize)
      module:init('bias', self.initialize)
   end
   -- Initialize the batch normalization layers
   for k,v in pairs(model:findModules('nn.BatchNormalization')) do
      v.weight:fill(1)
      v.bias:zero()
   end
   return model
end

function modelLSTM:weightsTransfer(model, trainedLayers)
   -- Find both LSTM and linear modules
   linearNodes = model:findModules('nn.Linear')
   -- Current linear layer
   local curLayer = 1
   for l = 1,#trainedLayers do
      -- Find equivalent in pre-trained layer
      lstmNodes = trainedLayers[l].encoder:findModules('nn.Linear')
      for k = 1,#lstmNodes do
	 linearNodes[curLayer].weights = lstmNodes[k].weight
	 linearNodes[curLayer].bias = lstmNodes[k].bias
	 curLayer = curLayer + 1
      end
   end
   return model
end

function modelLSTM:parametersDefault()
   self.initialize = nninit.kaiming
   self.addNonLinearity = true
   self.layerwiseLinear = true
   self.nonLinearity = nn.ReLU
   self.batchNormalize = true
   self.initForget = true
   self.sequencer = true
   self.pretrain = true
   self.dropout = 0.2
   self.windowSize = 16
   self.windowStep = 1
   self.rho = 5
end

function modelLSTM:loadParameters(options)
   local options = options or {}
   for option, value in pairs(options) do
      self[option] = value
   end
end

function modelLSTM:parametersRandom()
   -- All possible non-linearities
   self.distributions = {}
   self.distributions.nonLinearity = {nn.HardTanh, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU,
   -- nn.HardShrink, 
   }
   self.distributions.nonLinearityTest = {nn.LeakyReLU}
   self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse}
end

--[[
List of hyper-parameters:

SlidingWindow:  -> Moved to training script, messes up error computation if done inside
   of the network
  X size
  X step

Topology:
  X nLayers -> handled in an external loop
  * layers [1 … N]

Model:
  * rho
  * layerwiseLinear (boolean)
  * batchNormalize (boolean)
  X addNonLinearity (boolean)
  * nonLinearity (categorical (non linearities, e.g. ReLU))
  * dropout (real, [0, 1])

Learning:
  _ optimizationAlgorithm (categorical)
  _ learningRate (real)
  (+ algorithm-specific ?)
--]]

-- nn.HardShrink removed from this because it caused bugs 
local nonLinearityNames = {'nn.HardTanh', 'nn.SoftShrink', 'nn.SoftMax', 'nn.SoftMin', 'nn.SoftPlus', 'nn.SoftSign', 'nn.LogSigmoid', 'nn.LogSoftMax', 'nn.Sigmoid', 'nn.Tanh', 'nn.ReLU', 'nn.PReLU', 'nn.RReLU', 'nn.ELU', 'nn.LeakyReLU'}

-- Simple pattern-matching on the various names of non-linearities
-- 
-- Saving names rather than directly functions in the hyper-parameters is required
-- to be able to preperly retrieve the non-linearity used when outputting the
-- results.
local function getNonLinearity(name)
   if name == 'nn.HardTanh' then return nn.HardTanh 
   elseif name == 'nn.HardShrink' then return nn.HardShrink 
   elseif name == 'nn.SoftShrink' then return nn.SoftShrink 
   elseif name == 'nn.SoftMax' then return nn.SoftMax 
   elseif name == 'nn.SoftMin' then return nn.SoftMin 
   elseif name == 'nn.SoftPlus' then return nn.SoftPlus 
   elseif name == 'nn.SoftSign' then return nn.SoftSign 
   elseif name == 'nn.LogSigmoid' then return nn.LogSigmoid
   elseif name == 'nn.LogSoftMax' then return nn.LogSoftMax 
   elseif name == 'nn.Sigmoid' then return nn.Sigmoid 
   elseif name == 'nn.Tanh' then return nn.Tanh 
   elseif name == 'nn.ReLU' then return nn.ReLU
   elseif name == 'nn.PReLU' then return nn.PReLU 
   elseif name == 'nn.RReLU' then return nn.RReLU 
   elseif name == 'nn.ELU' then return nn.ELU
   elseif name == 'nn.LeakyReLU' then return nn.LeakyReLU
   else error('Unexpected non-linearity')
   end
end

local initializerNames = {'nninit.normal', 'nninit.uniform', 'nninit.xavier', 'nninit.kaiming', 'nninit.orthogonal', 'nninit.sparse'}

local function getInitializer(name)
   if name == 'nninit.normal' then return nninit.normal 
   elseif name == 'nninit.uniform' then return nninit.uniform 
   elseif name == 'nninit.xavier' then return nninit.xavier 
   elseif name == 'nninit.kaiming' then return nninit.kaiming
   elseif name == 'nninit.orthogonal' then return nninit.uniform 
   elseif name == 'nninit.sparse' then return nninit.sparse 
   else error('Unexpected initializer')
   end
end

-- Register non layer-specific parameters
function modelLSTM:registerOptions(hyperParams)
   hyperParams:registerParameter('rho', 'int', {1, 8})
   hyperParams:registerParameter('dropout', 'real', {0, 0.8})
   hyperParams:registerParameter('layerwiseLinear', 'bool')
   hyperParams:registerParameter('batchNormalize', 'bool')
   hyperParams:registerParameter('nonLinearity', 'catStr', nonLinearityNames)
   hyperParams:registerParameter('initializer', 'catStr', initializerNames)
end

-- Register non layer-specific parameters
function modelLSTM:updateOptions(hyperParams, optimizeBatchNormalize)
   self.rho = hyperParams:getCurrentParameter('rho');
   self.dropout = hyperParams:getCurrentParameter('dropout')
   self.layerwiseLinear = hyperParams:getCurrentParameter('layerwiseLinear')
   self.batchNormalize = hyperParams:getCurrentParameter('batchNormalize')
   self.nonLinearity = getNonLinearity(
      hyperParams:getCurrentParameter('nonLinearity'))
   self.initializer = getInitializer(
      hyperParams:getCurrentParameter('initializer'))
end

function modelLSTM:registerStructure(hyperParams, nLayers, minSize, maxSize)
   local minSize = minSize or 32
   local maxSize = maxSize or 4096
   for l = 1,nLayers do
    hyperParams:registerParameter("layer_" .. l, 'int', {minSize, maxSize});
  end
end

function modelLSTM:extractStructure(hyperParams, structure)
  structure.layers = {};
  for l = 1,structure.nLayers do
    structure.layers[l] = hyperParams:getCurrentParameter("layer_" .. l);
  end
  return structure
end
