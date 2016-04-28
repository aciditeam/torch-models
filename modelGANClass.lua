----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'nn'
require 'nngraph'
require 'unsup'
require 'optim'
require 'torch'
require 'modelGAN'
local nninit = require 'nninit'

local modelGANClass, parent = torch.class('modelGANClass', 'modelGAN')

function modelGANClass:defineClassifier(structure, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- Container:
  local model = nn.Sequential();
  local curTPoints = structure.nInputs;
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
      model:add(nn.Reshape(curTPoints * outSize)); model:add(nn.BatchNormalization(curTPoints * outSize)); model:add(nn.Reshape(curTPoints, outSize))
      curTPoints = curTPoints / structure.poolSize[i];
    end
    -- Non-linearity
    model:add(self.nonLinearity())
    -- Add dropout
    model:add(nn.Dropout(self.dropout));
    -- Pooling
    model:add(nn.TemporalMaxPooling(structure.poolSize[i], structure.poolSize[i]));
  end
  -- Reshape the outputs
  convOut = structure.convSize[#structure.convSize] * (structure.nInputs / torch.Tensor(structure.poolSize):cumprod()[#structure.poolSize]);
  model:add(nn.Reshape(convOut))
  -- Add final layers for regression
  model:add(nn.Linear(convOut, 1024))
  model:add(self.nonLinearity())
  model:add(nn.Dropout(self.dropout))
  -- Final layer takes a binary fake/real decision
  model:add(nn.Linear(1024,structure.nOutputs));
  return model;
end

function modelGANClass:defineGenerator(structure, options)
  local model = nn.Sequential();
  model:add(nn.JoinTable(2, 2));
  model:add(nn.Linear(self.noiseDim + structure.nOutputs, structure.nInputs / 2))
  model:add(nn.Reshape(structure.nInputs / 8, 4))
  model:add(self.nonLinearity())
  --model:add(nn.SpatialUpSamplingNearest(2))
  model:add(nn.Padding(2, -2)); model:add(nn.Padding(2, 2));
  model:add(nn.TemporalConvolution(4, 256, 5, 1, 2))
  model:add(nn.Reshape(structure.nInputs / 8 * 256))
  model:add(nn.BatchNormalization(structure.nInputs / 8 * 256))
  model:add(nn.Reshape(structure.nInputs / 8, 256))
  model:add(self.nonLinearity())
  --model:add(nn.SpatialUpSamplingNearest(2))
  model:add(nn.Padding(2, -2)); model:add(nn.Padding(2, 2));
  model:add(nn.TemporalConvolution(256, 256, 5, 1, 2))
  model:add(nn.Reshape(structure.nInputs / 8 * 256))
  model:add(nn.BatchNormalization(structure.nInputs / 8 * 256))
  model:add(nn.Reshape(structure.nInputs / 8, 256))
  model:add(self.nonLinearity())
  --model:add(nn.SpatialUpSamplingNearest(2))
  model:add(nn.Padding(2, -2)); model:add(nn.Padding(2, 2));
  model:add(nn.TemporalConvolution(256, 128, 5, 1, 2))
  model:add(nn.Reshape(structure.nInputs / 8 * 128))
  model:add(nn.BatchNormalization(structure.nInputs / 8 * 128))
  model:add(nn.Reshape(structure.nInputs / 8, 128))
  model:add(self.nonLinearity())
  model:add(nn.Padding(2, -1)); model:add(nn.Padding(2, 1));
  model:add(nn.TemporalConvolution(128, structure.nInputs, 3, 1, 1))
  model:add(nn.Reshape(structure.nInputs / 8 * 128));
  model:add(nn.Linear(structure.nInputs / 8 * 128, structure.nInputs))
  model:add(nn.View(structure.nInputs));
  return model
end

----------------------------------------------------------------------
-- Global Generalized Adversarial Network
----------------------------------------------------------------------
function modelGANClass:defineModel(structure, options)
  model = {};
  -- set the noise input to match the size
  self.noiseDim = (structure.nInputs / 8);
  -- define classifier network to train
  model.classifier = self:defineClassifier(structure, options);
  local tmpDiscriminator = model.classifier:clone('weight','bias', 'gradWeight','gradBias');
  -- define discriminator network to train
  tmpDiscriminator:insert(nn.JoinTable(2, 2), 1);
  tmpDiscriminator:insert(nn.Linear(structure.nInputs + structure.nOutputs, structure.nInputs), 2);
  --model.discriminator:add(tmpDiscriminator);
  tmpDiscriminator:add(nn.Linear(structure.nOutputs, 1));
  tmpDiscriminator:add(nn.Sigmoid());
  model.discriminator = tmpDiscriminator;
  -- define generator network to train
  model.generator = self:defineGenerator(structure, options);
  -- At model definition, both should be optimized
  self.optimize_D = true; self.optimize_G = true
  return model;
end

function modelGANClass:defineCriterion(model)
  -- loss function: negative log-likelihood
  loss = nn.BCECriterion();
  return model, loss
end

function modelGANClass:definePretraining(structure, l, options)
  -- TODO
  return model;
end

function modelGANClass:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder
  encoder:remove();
  return model.encoder;
end

function modelGANClass:getParameters(model)
  -- retrieve discriminator and generator
  model_D = model.discriminator;
  model_G = model.generator;
  model_C = model.classifier;
  -- retrieve parameters and gradients
  parameters_D,gradParameters_D = model_D:getParameters()
  parameters_G,gradParameters_G = model_G:getParameters()
  parameters_C,gradParameters_C = model_C:getParameters()
  -- create states for classifier
  sgdState_C = {
    learningRate = options.learningRate,
    momentum = options.momentum,
    optimize=true,
    numUpdates=0
  }
  -- only return nils (for compatibility)
  return nil, nil
end

function modelGANClass:supervisedTrain(model, trainData, options)
  -- retrieve discriminator and generator
  model_D = model.discriminator;
  model_G = model.generator;
  model_C = model.classifier;
  -- training function
  model_G:training()
  model_D:training()
  model_C:training()
  -- current epoch
  epoch = epoch or 1
  confusionGAN = optim.ConfusionMatrix({'Real','Fake'});
  -- keep track of time
  local time = sys.clock()
  -- shuffle order at each epoch
  shuffle = torch.randperm(trainData.data:size(1));
  -- do one epoch
  print("==> epoch # " .. epoch .. ' [batch = ' .. options.batchSize .. ']')
  for t = 1,trainData.data:size(1),options.batchSize do
    -- disp progress
    xlua.progress(t, trainData.data:size(1))
    -- Check size (for last batch)
    bSize = math.min(options.batchSize, trainData.data:size(1) - t + 1)
    -- The data batch (coming from dataset) will only be half real images
    local dataBatchSize = bSize / 2; 
    -- create mini batch
    if (trainData.data[1]:nDimension() == 1) then
      inputs = torch.Tensor(bSize, trainData.data[1]:size(1))
    else
      inputs = torch.Tensor(bSize, trainData.data[1]:size(1), trainData.data[1]:size(2))
    end
    local targets = torch.zeros(bSize)
    local labels = torch.zeros(bSize)
    -- Will be used to update the generator
    local noise_inputs = torch.Tensor(bSize, self.noiseDim);
    -- Used to condition on classes
    local class_inputs = torch.zeros(bSize, trainData.labels:max());
    local k = 1;
    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      -- get current parameters
      if x ~= parameters_D then 
        parameters_D:copy(x)
      end
      -- reset gradients
      gradParameters_D:zero() 
      --  forward pass
      local outputs = model_D:forward({inputs, class_inputs})
      -- Separate errors between reals (R) and fakes (F)
      err_R = criterion:forward(outputs:narrow(1, 1, bSize / 2), targets:narrow(1, 1, bSize / 2))
      err_F = criterion:forward(outputs:narrow(1, (bSize / 2) + 1, bSize / 2), targets:narrow(1, (bSize / 2) + 1, bSize / 2))
      -- Learning margin
      local margin = self.margin
      -- Need to balance learning between the two learners
      self.optimize_D = true; self.optimize_G = true
      -- Stop the discriminator learning if the recognition is too strong (very low errors)
      if err_F < margin or err_R < margin then self.optimize_D = false; end
      -- Stop the genertor learning if the recognition is too weak (very high errors)
      if err_F > (1.0-margin) or err_R > (1.0-margin) then self.optimize_G = false; end
      -- If both have been deactivated, no learning happens ! Reset it
      if self.optimize_D == false and self.optimize_G == false then self.optimize_G = true; self.optimize_D = true; end
      io.write("v1_lfw| R:", err_R,"  F:", err_F, "  ")
      -- forward through the criterion
      local f = criterion:forward(outputs, targets)
      -- backward pass
      local df_do = criterion:backward(outputs, targets)
      model_D:backward({inputs, class_inputs}, df_do);
      -- penalties (L1 and L2):
      if options.regularizeL1 ~= 0 or options.regularizeL2 ~= 0 then
        local norm, sign= torch.norm,torch.sign
        -- Loss:
        f = f + options.regularizeL1 * norm(parameters_D,1)
        f = f + options.regularizeL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(options.regularizeL1) + parameters_D:clone():mul(options.regularizeL2) )
      end
      -- update confusion (add 1 since targets are binary)
      for i = 1,bSize do
        local c
        if outputs[i][1] > 0.5 then c = 2 else c = 1 end
        confusionGAN:add(c, targets[i]+1)
      end
      return f,gradParameters_D
    end
    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of classifier 
    local fevalC = function(x)
      collectgarbage()
      -- get new parameters
      if x ~= parameters_G then 
        parameters_C:copy(x)
      end
      -- reset gradients
      gradParameters_C:zero() 
      -- forward only the real inputs through the classifier
      local outputs = model_C:forward(inputs[{{1,dataBatchSize}}])
      tmpCrit = nn.ClassNLLCriterion()
      -- forward through the criterion
      local f = tmpCrit:forward(outputs, labels[{{1,dataBatchSize}}])
      -- update confusion
      for i = 1,outputs:size(1) do
        confusion:add(outputs[i], labels[i]) 
      end
      -- backward pass
      local df_samples = tmpCrit:backward(outputs, labels[{{1,dataBatchSize}}])
      -- backward through the classifier
      model_C:backward(inputs[{{1,dataBatchSize}}], df_samples);
      -- update the second confusion matrix
      return f,gradParameters_C
    end
    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      -- get new parameters
      if x ~= parameters_G then 
        parameters_G:copy(x)
      end
      -- reset gradients
      gradParameters_G:zero() 
      -- forward through generator (get fake samples)
      local samples = model_G:forward({noise_inputs, class_inputs})
      -- forward fake samples through the discriminator
      local outputs = model_D:forward({samples, class_inputs})
      -- forward through the criterion
      local f = criterion:forward(outputs, targets)
      -- backward pass
      local df_samples = criterion:backward(outputs, targets)
      -- need to backward through the discriminator as well !
      model_D:backward({samples, class_inputs}, df_samples)
      -- then propagate the gradient through the generator
      local df_do = model_D.gradInput[1]
      model_G:backward({noise_inputs, class_inputs}, df_do)
      return f,gradParameters_G
    end
    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    -- Get half a minibatch of real, half fake
    for k=1,self.K do
      -- (1.1) Real data 
      local k = 1
      -- iterate over mini-batch examples
      for i = t,math.min(t+dataBatchSize-1,trainData.data:size(1)) do
        local idx = math.random(trainData.data:size(1))
        local sample = trainData.data[idx]
        inputs[k] = sample:clone()
        labels[k] = trainData.labels[idx];
        class_inputs[k][trainData.labels[idx]] = 1;
        k = k + 1
      end
      targets[{{1,dataBatchSize}}]:fill(1)
      -- (1.2) Sampled data
      noise_inputs:normal(0, 1)
      local tmpClass = torch.zeros(dataBatchSize, trainData.labels:max());
      for i = 1, dataBatchSize do
        local idx = math.random(trainData.data:size(1))
        tmpClass[i][trainData.labels[idx]] = 1;
      end
      local samples = model_G:forward({noise_inputs[{{dataBatchSize+1,bSize}}], tmpClass})
      for i = 1, dataBatchSize do
        inputs[k] = samples[i]:clone()
        class_inputs[k] = tmpClass[i];
        k = k + 1
      end
      targets[{{dataBatchSize+1,bSize}}]:fill(0)
      -- First optimize the discriminator
      optimState.optimize = self.optimize_D;
      optimMethod(fevalD, parameters_D, optimState)
      -- Then optimize the classifier
      optimMethod(fevalC, parameters_C, sgdState_C)
    end
    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    -- Generate a set of noise inputs
    noise_inputs:normal(0, 1)
    -- All targets are fake
    targets:fill(1)
    -- Still fill with random classes
    for i = 1,bSize do
      local idx = math.random(trainData.labels:size(1))
      class_inputs[i][trainData.labels[idx]] = 1;
    end
    optimState.optimize = self.optimize_G;
    sgdState_G = {
      learningRate = options.learningRate,
      momentum = options.momentum,
      optimize=self.optimize_G,
      numUpdates=0
    }
    optimMethod(fevalG, parameters_G, sgdState_G)
    -- display progress
    xlua.progress(t, trainData.data:size(1))
  end -- end for loop over dataset
  -- time taken
  time = sys.clock() - time
  time = time / trainData.data:size(1)
  print("time to learn 1 sample = " .. (time*1000) .. 'ms')
  -- print confusion matrix
  print('Adversarial confusion :')
  print(confusionGAN)
  print('Class confusion :')
  print(confusion)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  -- next epoch
  epoch = epoch + 1
  return (1 - confusion.totalValid);
end

function modelGANClass:supervisedTest(model, testData, options)
  return 1;
end

function modelGANClass:weightsTransfer(model, trainedLayers)
  -- TODO
  return model;
end

function modelGANClass:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = false;
  self.pretrainType = 'ae';
  self.pretrain = false;
  self.padding = true;
  self.dropout = 0.25;
  self.margin = 0.3;
  self.K = 1;
end