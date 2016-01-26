----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'unsup'
require 'optim'
require 'torch'

modelRNN = {};

----------------------------------------------------------------------
-- Handmade RNN
-- (From the Oxford course)
----------------------------------------------------------------------
function modelRNN.defineRNN(input_size, rnn_size, params)
  local n = params.n or 1
  local dropout = params.dropout or 0
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]

  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    
    local prev_h = inputs[L+1]
    if L == 1 then 
      --x = OneHot(input_size)(inputs[1])
      x = inputs[1];
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

--[[

--
-- RNN CLASSIFICATION
--

require 'rnn'

batchSize = 10
rho = 5
-- used to call the BPTT
updateInterval = 4
hiddenSize = 32
nIndex = 10000

-- Model
model = nn.Sequential()
model:add(nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize),
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
   rho
))
model:add(nn.Linear(hiddenSize, nIndex))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

-- dummy dataset (task is to predict next item, given previous)
dataset = torch.randperm(nIndex)

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)

function gradientUpgrade(model, x, y, criterion, learningRate, i)
  local prediction = model:forward(x)
  local err = criterion:forward(prediction, y)
  local gradOutputs = criterion:backward(prediction, y)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   model:backward(x, gradOutputs)

   if i % 100 == 0 then
      print('error for iteration ' .. i  .. ' is ' .. err/rho)
   end

   if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
      model:backwardThroughTime()
      model:updateParameters(learningRate)
      model:zeroGradParameters()
   end
end


lr = 0.01
for i = 1, 10e4 do
   local inputs = dataset:index(1, offsets)
   -- shift of one batch indexes
   offsets:add(1)
   for j=1,batchSize do
      if offsets[j] > nIndex then
         offsets[j] = 1
      end
   end
   local targets = dataset:index(1, offsets)
   gradientUpgrade(model, inputs, targets, criterion, lr, i)
end

--
-- RNN SEQUENCER CLASS
--

As an example, let's use Sequencer and Recurrence to build a Simple RNN for language modeling :

rho = 5
hiddenSize = 10
outputSize = 5 -- num classes
nIndex = 10000

-- recurrent module
rm = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.LookupTable(nIndex, hiddenSize))
      :add(nn.Linear(hiddenSize, hiddenSize)))
   :add(nn.CAddTable())
   :add(nn.Sigmoid())

rnn = nn.Sequencer(
   nn.Sequential()
      :add(nn.Recurrence(rm, hiddenSize, 1))
      :add(nn.Linear(hiddenSize, outputSize))
      :add(nn.LogSoftMax())
)

require 'rnn'

batchSize = 50
rho = 5
hiddenSize = 12
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


--rnn layer
rnn = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize),
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
   rho
)

-- Model
model = nn.Sequential()
model:add(nn.Sequencer(rnn))
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

lr = 0.01
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

   i = gradientUpgrade(model, inputs, targets, criterion, lr, i)
end

require 'rnn'

batchSize = 8
rho = 5
hiddenSize = 10
nIndex = 10000
-- RNN
r = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, nIndex))
rnn:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

-- dummy dataset (task is to predict next item, given previous)
sequence = torch.randperm(nIndex)

offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*batchSize))
end
offsets = torch.LongTensor(offsets)

lr = 0.1
updateInterval = 4
i = 1
while true do
   -- a batch of inputs
   local input = sequence:index(1, offsets)
   local output = rnn:forward(input)
   -- incement indices
   offsets:add(1)
   for j=1,batchSize do
      if offsets[j] > nIndex then
         offsets[j] = 1
      end
   end
   local target = sequence:index(1, offsets)
   local err = criterion:forward(output, target)
   print(err)
   local gradOutput = criterion:backward(output, target)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   rnn:backward(input, gradOutput)

   i = i + 1
   -- note that updateInterval < rho
   if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      rnn:backwardThroughTime()
      -- 2. updates parameters
      rnn:updateParameters(lr)
      rnn:zeroGradParameters()
      -- 3. reset the internal time-step counter
      rnn:forget()
   end
end


EXAMPLE FOR TIME SERIES !!!

-- This is a modification of an example provided at https://github.com/Element-Research/rnn#rnn.Recurrent
-- Please refer to https://github.com/tindzk/rnntest01 for a detailed example
require 'rnn'

--batchSize = 1
rho = 5
hiddenSize = 10
-- RNN
r = nn.Recurrent(
   hiddenSize, --size of the input layer
   nn.Linear(1, hiddenSize), --input layer
   nn.Linear(hiddenSize, hiddenSize), --recurrent layer
   nn.Sigmoid(), --transfer function
   rho  --maximum number of time steps for BPTT
)

rnn = nn.Sequential()
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, 1))

criterion = nn.MSECriterion() 

-- dummy dataset (task is to predict next item, given previous)
i = 0
sequence = torch.Tensor(10):apply(function() --fill with a simple arithmetic progression 0.1, 0.2, .. 0.9, 1, 0.1,..
  i = i + 0.1
  if i >1 then i = 0 end
  return i
end)
print('Sequence:')
print(sequence)

lr = 0.1
step = 0
threshold = 0.002
thresholdStep = 0
--while true do
for k = 1, 100 do
for j = 1, sequence:size(1)-1 do
   step = step + 1
   -- a batch of inputs
   local input = torch.Tensor(1):fill(sequence[j])
   local output = rnn:forward(input)
   local target = torch.Tensor(1):fill(sequence[j+1]) --target is the next numbet in sequence
   local err = criterion:forward(output, target)
   print('Step: ', step, ' Input: ', input[1], ' Target: ', target[1], ' Output: ', output[1], ' Error: ', err)
   if (err < threshold and thresholdStep == 0) then thresholdStep = step end --remember this step
   local gradOutput = criterion:backward(output, target)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   rnn:backward(input, gradOutput)
   
   -- note that updateInterval < rho
   if j % 3 == 0 then --update interval
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      rnn:backwardThroughTime()
      -- 2. updates parameters
      rnn:updateParameters(lr)
      rnn:zeroGradParameters()
      -- 3. reset the internal time-step counter
      rnn:forget()
   end --end if
end -- end j
end -- end k

print('Error < ', threshold,' on step: ', thresholdStep)

BI-DIRECTIONAL RNN :

Applies encapsulated fwd and bwd rnns to an input sequence in forward and reverse order. It is used for implementing Bidirectional RNNs and LSTMs.

brnn = nn.BiSequencer(fwd, [bwd, merge])
The input to the module is a sequence (a table) of tensors and the output is a sequence (a table) of tensors of the same length. Applies a fwd rnn (an AbstractRecurrent instance) to each element in the sequence in forward order and applies the bwd rnn in reverse order (from last element to first element). The bwd rnn defaults to:

bwd = fwd:clone()
bwd:reset()
For each step (in the original sequence), the outputs of both rnns are merged together using the merge module (defaults to nn.JoinTable(1,1)). If merge is a number, it specifies the JoinTable constructor's nInputDim argument. Such that the merge module is then initialized as :

merge = nn.JoinTable(1,merge)

-- Model

-- language model
lm = nn.Sequential()

local inputSize = opt.hiddenSize[1]
for i,hiddenSize in ipairs(opt.hiddenSize) do 

   if i~= 1 and not opt.lstm then
      lm:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize)))
   end
   
   -- recurrent layer
   local rnn
   if opt.lstm then
      -- Long Short Term Memory
      rnn = nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize))
   else
      -- simple recurrent neural network
      rnn = nn.Recurrent(
         hiddenSize, -- first step will use nn.Add
         nn.Identity(), -- for efficiency (see above input layer) 
         nn.Linear(hiddenSize, hiddenSize), -- feedback layer (recurrence)
         nn.Sigmoid(), -- transfer function 
         99999 -- maximum number of time-steps per sequence
      )
      if opt.zeroFirst then
         -- this is equivalent to forwarding a zero vector through the feedback layer
         rnn.startModule:share(rnn.feedbackModule, 'bias')
      end
      rnn = nn.Sequencer(rnn)
   end

   lm:add(rnn)
   
   if opt.dropout then -- dropout it applied between recurrent layers
      lm:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   end
   
   inputSize = hiddenSize
end

if opt.bidirectional then
   -- initialize BRNN with fwd, bwd RNN/LSTMs
   local bwd = lm:clone()
   bwd:reset()
   bwd:remember('neither')
   local brnn = nn.BiSequencerLM(lm, bwd)
   
   lm = nn.Sequential()
   lm:add(brnn)
   
   inputSize = inputSize*2
end

-- input layer (i.e. word embedding space)
lm:insert(nn.SplitTable(1,2), 1) -- tensor to table of tensors

if opt.dropout then
   lm:insert(nn.Dropout(opt.dropoutProb), 1)
end

lookup = nn.LookupTable(ds:vocabularySize(), opt.hiddenSize[1], opt.accUpdate)
lookup.maxOutNorm = -1 -- disable maxParamNorm on the lookup table
lm:insert(lookup, 1)

-- output layer
if opt.softmaxforest or opt.softmaxtree then
   -- input to nnlm is {inputs, targets} for nn.SoftMaxTree
   local para = nn.ParallelTable()
   para:add(lm):add(opt.cuda and nn.Sequencer(nn.Convert()) or nn.Identity())
   lm = nn.Sequential()
   lm:add(para)
   lm:add(nn.ZipTable())
   if opt.softmaxforest then -- requires a lot more memory
      local trees = {ds:hierarchy('word_tree1.th7'), ds:hierarchy('word_tree2.th7'), ds:hierarchy('word_tree3.th7')}
      local rootIds = {880542,880542,880542}
      softmax = nn.SoftMaxForest(inputSize, trees, rootIds, opt.forestGaterSize, nn.Tanh(), opt.accUpdate)
      opt.softmaxtree = true
   elseif opt.softmaxtree then -- uses frequency based tree
      local tree, root = ds:frequencyTree()
      softmax = nn.SoftMaxTree(inputSize, tree, root, opt.accUpdate)
   end
else
   if #ds:vocabulary() > 50000 then
      print("Warning: you are using full LogSoftMax for last layer, which "..
         "is really slow (800,000 x outputEmbeddingSize multiply adds "..
         "per example. Try --softmaxtree instead.")
   end
   softmax = nn.Sequential()
   softmax:add(nn.Linear(inputSize, ds:vocabularySize()))
   softmax:add(nn.LogSoftMax())
end
lm:add(nn.Sequencer(softmax))

if opt.uniform > 0 then
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

if opt.dataset ~= 'BillionWords' then
   -- will recurse a single continuous sequence
   lm:remember(opt.lstm and 'both' or 'eval')
end
   

-- Propagators
if opt.lrDecay == 'adaptive' then
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
   opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

train = dp.Optimizer{
   loss = opt.softmaxtree and nn.SequencerCriterion(nn.TreeNLLCriterion())
         or nn.ModuleCriterion(
            nn.SequencerCriterion(nn.ClassNLLCriterion()), 
            nn.Identity(), 
            opt.cuda and nn.Sequencer(nn.Convert()) or nn.Identity()
         ),
   epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         if opt.lrDecay == 'adaptive' then
            opt.learningRate = opt.learningRate*ad.decay
            ad.decay = 1
         elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
            opt.learningRate = opt.schedule[report.epoch]
         elseif opt.lrDecay == 'linear' then 
            opt.learningRate = opt.learningRate + opt.decayFactor
         end
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
            if opt.meanNorm then
               print("mean gradParam norm", opt.meanNorm)
            end
         end
      end
   end,
   callback = function(model, report) -- called every batch
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         if opt.cutoffNorm > 0 then
            local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
            opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
         end
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Perplexity(),  
   sampler = torch.isTypeOf(ds, 'dp.TextSource')
      and dp.TextSampler{epoch_size = opt.trainEpochSize, batch_size = opt.batchSize}
      or dp.RandomSampler{epoch_size = opt.trainEpochSize, batch_size = opt.batchSize}, 
   acc_update = opt.accUpdate,
   progress = opt.progress
}

if not opt.trainOnly then
   valid = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = torch.isTypeOf(ds, 'dp.TextSource') 
         and dp.TextSampler{epoch_size = opt.validEpochSize, batch_size = 1} 
         or dp.SentenceSampler{epoch_size = opt.validEpochSize, batch_size = 1, max_size = 100},
      progress = opt.progress
   }
   tester = dp.Evaluator{
      feedback = dp.Perplexity(),  
      sampler = torch.isTypeOf(ds, 'dp.TextSource') 
         and dp.TextSampler{batch_size = 1} 
         or dp.SentenceSampler{batch_size = 1, max_size = 100}  -- Note : remove max_size for exact test set perplexity (will cost more memory)
   }
end

-- Experiment
xp = dp.Experiment{
   model = lm,
   optimizer = train,
   validator = valid,
   tester = tester,
   observer = {
      ad,
      dp.FileLogger(),
      dp.EarlyStopper{
         max_epochs = opt.maxTries, 
         error_report={opt.trainOnly and 'optimizer' or 'validator','feedback','perplexity','ppl'}
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch,
   target_module = nn.SplitTable(1,1):type('torch.IntTensor')
}
if opt.softmaxtree then
   -- makes it forward {input, target} instead of just input
   xp:includeTarget()
end

-- GPU or CPU
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   if opt.softmaxtree or opt.softmaxforest then
      require 'cunnx'
   end
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Language Model :"
   print(lm)
end

xp:run(ds)

--]]