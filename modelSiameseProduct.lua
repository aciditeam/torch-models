----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Siamese network
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'torch'
require 'nn'
local nninit = require 'nninit'

local modelSiameseProduct, parent = torch.class('modelSiameseProduct', 'modelSiamese')

function modelSiameseProduct:defineModel(structure, options)
  -- Define a single network
  local encoder = self:defineSimple(structure, options);
  -- Turn it into a siamese model (input splits on 1st dimension)
  local siamese_encoder = nn.ParallelTable()
  siamese_encoder:add(encoder)
  -- Clone the encoder and share the weight, bias (must also share the gradWeight and gradBias)
  siamese_encoder:add(encoder:clone('weight', 'bias', 'gradWeight', 'gradBias'))
  -- Enclose the first siamese parallel (for dot product)
  local dotEncoder = nn.Sequential()
  dotEncoder:add(siamese_encoder)
  dotEncoder:add(nn.DotProduct())
  local dotClone = dotEncoder:clone('weight', 'bias', 'gradWeight', 'gradBias');
  -- The siamese model (of siamese models !)
  local dotParallel = nn.ParallelTable()
  dotParallel:add(dotEncoder);
  dotParallel:add(dotClone);
  -- Create a 3rd pathway for a classifier
  local classEncoder = encoder:clone('weight','bias', 'gradWeight','gradBias');
  -- Add a linear layer for logistic regression
  classEncoder:add(nn.Linear(structure.nOutputs, structure.nOutputs));
  classEncoder:add(nn.LogSoftMax())
  -- Final full model
  local fullModel = nn.ParallelTable();
  fullModel:add(dotParallel);
  fullModel:add(classEncoder);
  return fullModel
end

function modelSiameseProduct:defineCriterion(model)
  local margin = 1;
  local loss = nn.ParallelCriterion();
  --oss:add(nn.HingeEmbeddingCriterion(margin));
  loss:add(nn.MarginRankingCriterion(margin))
  loss:add(nn.ClassNLLCriterion());
  return model, loss;
end

-- Function to perform supervised training on the full model
function modelSiameseProduct:supervisedTrain(model, trainData, options)
  -- epoch tracker
  epoch = epoch or 1
  -- time variable
  local time = sys.clock()
  -- adjust the batch size (needs at least 2 examples)
  adjBSize = (options.batchSize > 1 and options.batchSize or 2)
  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training();
  -- shuffle order at each epoch
  shuffle = torch.randperm(trainData.data:size(1));
  -- do one epoch
  print("==> epoch # " .. epoch .. ' [batch = ' .. adjBSize .. ' (' .. ((adjBSize * (adjBSize - 1)) / 2) .. ')]')
  for t = 1,trainData.data:size(1),adjBSize do
    -- disp progress
    xlua.progress(t, trainData.data:size(1))
    -- Check size (for last batch)
    bSize = math.min(adjBSize, trainData.data:size(1) - t + 1)
    -- Real batch size is combinatorial
    combBSize = ((bSize * (bSize - 1)) / 2)
    -- Maximum indice to account
    mId = math.min(t+options.batchSize-1,trainData.data:size(1))
    -- create mini batch
    local inputs = {}
    local targets = {}
    local k = 1;
    -- iterate over mini-batch examples
    for i = t,(mId - 1) do
      -- load first sample
      local i1 = trainData.data[shuffle[i]]
      local t1 = trainData.labels[shuffle[i]]
      for j = i+1,mId do
        -- load second sample
        local i2 = trainData.data[shuffle[j]]
        local t2 = trainData.labels[shuffle[j]]
        -- Find a non-matching pair
        for l = t,mId do
          -- load second sample
          local i3 = trainData.data[shuffle[l]]
          local t3 = trainData.labels[shuffle[l]]
          if ((t1 == t2) and (t1 ~= t3)) then
            inputs[k] = {{{i1, i2}, {i1, i3}}, i1};
            targets[k] = {-1, t1};
            k = k + 1
          end
          if ((t1 ~= t2) and (t1 == t3)) then
            inputs[k] = {{{i1, i2}, {i1, i3}}, i1};
            targets[k] = {1, t1};
            k = k + 1
          end 
        end
      end
    end
    if options.type == 'double' then inputs = inputs:double() end
    if options.cuda then inputs = inputs:cuda() end
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end
      -- reset gradients
      gradParameters:zero()
      -- f is the average of all criterions
      local f = 0
      -- [[ Evaluate function for each example of the mini-batch ]]--
      for i = 1,#inputs do
        -- estimate forward pass
        local output = model:forward(inputs[i])
        -- estimate error (here compare to margin)
        local err = criterion:forward(output, targets[i])
        -- compute overall error
        f = f + err
        -- estimate df/dW (perform back-prop)
        local df_do = criterion:backward(output, targets[i])
        model:backward(inputs[i], df_do)
        -- in case of combined criterion
        output = output[2];
        -- update confusion
        confusion:add(output, targets[i][2])
      end
      -- Normalize gradients and error
      gradParameters:div(#inputs);
      f = f / #inputs
      -- return f and df/dX
      return f,gradParameters
    end
    -- optimize on current mini-batch
    optimMethod(feval, parameters, optimState)
  end
  -- time taken
  time = sys.clock() - time;
  time = time / trainData.data:size(1);
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
  -- print confusion matrix
  print(confusion)
  -- update logger/plot
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  if options.plot then
    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    trainLogger:plot()
  end
  -- save/log current net
  local filename = paths.concat(options.save, 'model.net')
  --os.execute('mkdir -p ' .. sys.dirname(filename))
  --torch.save(filename, model)
  -- next epoch
  epoch = epoch + 1
  return (1 - confusion.totalValid);
end

------------------------------------------
--
-- Function to perform supervised testing on the model
--
------------------------------------------
function modelSiameseProduct:supervisedTest(modelOrig, testData, options)
  -- local vars
  local time = sys.clock()
  -- adjust the batch size (needs at least 2 examples)
  adjBSize = (options.batchSize > 1 and options.batchSize or 2)
  local model = modelOrig:get(2);
  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  model:evaluate();
  -- test over test data
  print('==> testing on test set:')
  for t = 1,testData.data:size(1),options.batchSize do
    -- disp progress
    xlua.progress(t, testData.data:size(1))
    -- Check size (for last batch)
    bSize = math.min(adjBSize, testData.data:size(1) - t + 1)
    -- Maximum indice to account
    mId = math.min(t+options.batchSize-1,testData.data:size(1))
    -- create mini batch
    local inputs = {}
    local targets = {}
    local k = 1;
    -- iterate over mini-batch examples
    for i = t,mId do
      -- load first sample
      local i1 = testData.data[i]
      local t1 = testData.labels[i]
      inputs[k] = i1;
      targets[k] = t1;
      k = k + 1
    end
    -- test sample
    for i = 1,#inputs do
      -- Predict class and embedding
      local pred = model:forward(inputs[i])
      -- Here we have a combined criterion
      confusion:add(pred, targets[i])
    end
  end
  -- timing
  time = sys.clock() - time
  time = time / testData.data:size(1)
  print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
  -- print confusion matrix
  print(confusion)
  -- update log/plot
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  if options.plot then
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    testLogger:plot()
  end
  -- averaged param use?
  if average then
    -- restore parameters
    parameters:copy(cachedparams)
  end
  -- next iteration:
  -- confusion:zero()
  return (1 - confusion.totalValid);
end

function modelSiameseProduct:definePretraining(structure, l, options)
  -- TODO
  return model;
end

function modelSiameseProduct:retrieveEncodingLayer(model)
  -- Here simply return the encoder
  encoder = model.encoder
  encoder:remove();
  return model.encoder;
end

function modelSiameseProduct:weightsInitialize(model)
  -- TODO
  return model;
end

function modelSiameseProduct:weightsTransfer(model, trainedLayers)
  -- TODO
  return model;
end

function modelSiameseProduct:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.pretrainType = 'ae';
  self.pretrain = false;
  self.dropout = 0.5;
end

function modelSiameseProduct:parametersRandom()
  -- All possible non-linearities
  self.distributions = {};
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
  self.distributions.batchNormalize = {true, false};
  self.distributions.pretrainType = {'ae', 'psd'};
  self.distributions.pretrain = {true, false};
  self.distributions.dropout = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
end