----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- The optime package contains several optimization routines for Torch. Most optimization algorithms has the following interface:
--
-- x*, {f}, ... = optim.method(opfunc, x, state)
-- 
-- opfunc : a user-defined closure that respects this API: f, df/dx = opfunc(x)
-- x      : the current parameter vector (a 1D torch.Tensor)
-- state  : a table of parameters, and state variables, dependent upon the algorithm
-- x*     : the new parameter vector that minimizes f, x* = argmin_x f(x)
-- {f}    : a table of all f values, in the order they've been evaluated (for some simple algorithms, like SGD, #f == 1)
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'torch'
require 'nn'
require 'xlua'
require 'optim'

----------------------------------------------------------------------
-- Perform  configuration of the optimizer
-- 
-- Rely on a optimization structure with options
--  . optimization  ('SGD')       
--  optimization methods: 
--  SGD | ASGD | LBFGS | CG | ADADELTA | ADAGRAD | ADAM | ADAMAX | FISTALS | NAG | RMSPROP | RPROP | CMAES
--  
--  Then parameters depend on the optimization method used
--  
--  Stochastic Gradient Descent (SGD)
--  . learningRate      (2e-3)        - learning rate at t=0
--  . learningRateDecay (1e-5)        - learning rate decay
--  . weightDecay       (1e-5)        - weight decay
--  . weightDecays      (nil)         - vector of individual weight decays
--  . momentum          (1e-1)        - gradient momentum
--  . dampening         (0)           - dampening for momentum
--  . nesterov          (true)        - enables Nesterov momentum
--  
--  Averaged Stochastic Gradient Descent (ASGD)
--  . eta0              (2e-3)        - learning rate at t=0
--  . lambda            (1e-5)        - learning rate decay
--  . alpha             (1)           - power for eta update
--  . t0                (10)          - point at which to start averaging
--  
--  Limited-Memory BFGS (LBFGS)
--  . learningRate      (2e-3)        - if no line search provided, then a fixed step size is used
--  . maxIter           (25)          - maximum number of iterations allowed
--  . maxEval           (50)          - maximum number of function evaluations
--  . tolFun            (1e-4)        - termination tolerance on the first-order optimality
--  . tolX              (1e-8)        - termination tol on progress in terms of func/param changes
--  . lineSearch        (lswolfe)     - a line search function
--  . nCorrection       (100)         - number of corrections
--  
--  Conjugate Gradient (CG)
--  . maxIter           (25)          - maximum number of iterations allowed
--  . maxEval           (50)          - maximum number of function evaluations
--  
--  Adaptative Delta (ADADELTA)
--  . rho               (0.9)         - interpolation parameter
--  . eps               (1e-6)        - for numerical stability
--  
--  Adaptative Gradient (ADAGRAD)
--  . learningRate      (2e-3)        - learning rate
--  . learningRateDecay (1e-5)        - learning rate decay
--  
--  ADAM (ADAM)
--  . learningRate      (2e-3)        - learning rate
--  . beta1             (0.9)         - first moment coefficient
--  . beta2             (0.999)       - second moment coefficient
--  . epsilon           (1e-8)        - numerical stability
--  
--  ADAMAX (ADAMAX)
--  . learningRate      (2e-3)        - learning rate
--  . beta1             (0.9)         - first moment coefficient
--  . beta2             (0.999)       - second moment coefficient
--  . epsilon           (1e-8)        - numerical stability
--  
--  Nesterov's Accelerated Gradient (NAG)
--  . learningRate      (2e-3)        - learning rate at t=0
--  . learningRateDecay (1e-5)        - learning rate decay
--  . weightDecay       (1e-5)        - weight decay
--  . weightDecays      (nil)         - vector of individual weight decays
--  . momentum          (1e-1)        - gradient momentum
--  
--  RMSProp (RMSPROP)
--  . learningRate      (2e-3)        - learning rate
--  . alpha             (0.99)        - first moment coefficient
--  . epsilon           (1e-8)        - value to initialize m
--  
--  RProp (RPROP)
--  . stepsize          ()            - initial step size, common to all components
--  . etaplus           (1.2)         - multiplicative increase factor, > 1
--  . etaminus          (0.5)         - multiplicative decrease factor, < 1
--  . stepsizemax       (50)          - maximum stepsize allowed
--  . stepsizemin       (1e-6)        - minimum stepsize allowed
--  . niter             (10)          - number of iterations
--
-- Covariance Matrix Adaptation Evolution Strategy (CMAES)
--  . sigma             (1e-3)        - initial step-size (standard deviation in each coordinate)
--  . maxEval           (25)          - maximal number of function evaluations
--  . ftarget           (1e-3)        - target function value (stop if fitness < ftarget)
--  . popsize           (nil)         - population size. If this is left empty, 4 + int(3 * log(|x|)) will be used
--
function configureOptimizer(options, dataSize)
   if options.optimization == 'SGD' then
      optimState = {
	 learningRate = options.learningRate or 2e-3,
	 learningRateDecay = options.learningRateDecay or 1e-5,
	 weightDecay = options.weightDecay or 1e-5,
	 weightDecays = options.weightDecays or nil,
	 momentum = options.momentum or 1e-1,
	 dampening = options.dampening or 0,
	 nesterov = options.nesterov or true
      }
      optimMethod = optim.sgd
   elseif options.optimization == 'ASGD' then
      optimState = {
	 eta0 = options.learningRate or 2e-3,
	 lambda = options.learningRateDecay or 1e-5,
	 alpha = options.alpha or 1,
	 t0 = dataSize * options.t0 or 10
      }
      optimMethod = optim.asgd
   elseif options.optimization == 'LBFGS' then
      optimState = {
	 learningRate = options.learningRate or 2e-3,
	 maxIter = options.maxIter or 25,
	 maxEval = options.maxEval or 50,
	 tolFun = options.tolFun or 1e-4,
	 tolX = options.tolX or 1e-8,
	 lineSearch = options.lineSearch or optim.lswolfe,
	 nCorrection = options.nCorrection or 100
      }
      optimMethod = optim.lbfgs
   elseif options.optimization == 'CG' then
      optimState = {
	 maxIter = options.maxIter or 25,
	 maxEval = options.maxEval or 50
      }
      optimMethod = optim.cg
   elseif options.optimization == 'ADADELTA' then
      optimState = {
	 rho = options.rho or 0.9,
	 eps = options.eps or 1e-6
      }
      optimMethod = optim.adadelta
   elseif options.optimization == 'ADAGRAD' then
      optimState = {
	 learningRate = options.learningRate or 2e-3,
	 learningRateDecay = options.learningRateDecay or 1e-5
      }
      optimMethod = optim.adagrad
   elseif options.optimization == 'ADAM' then
      optimState = {
	 learningRate = options.learningRate or 2e-3,
	 beta1 = options.beta1 or 0.9,
	 beta2 = options.beta2 or 0.999,
	 epsilon = options.epsilon or 1e-8
      }
      optimMethod = optim.adam
   elseif options.optimization == 'ADAMAX' then
      optimState = {
	 learningRate = options.learningRate or 2e-3,
	 beta1 = options.beta1 or 0.9,
	 beta2 = options.beta2 or 0.999,
	 epsilon = options.epsilon or 1e-8
      }
      optimMethod = optim.adamax
   elseif options.optimization == 'NAG' then
      optimState = {
	 learningRate = options.learningRate or 2e-3,
	 learningRateDecay = options.learningRateDecay or 1e-5,
	 weightDecay = options.weightDecay or 1e-5,
	 momentum = options.momentum or 1e-1
      }
      optimMethod = optim.nag
   elseif options.optimization == 'RMSPROP' then
      optimState = {
	 learningRate = options.learningRate or 2e-3,
	 alpha = options.alpha or 0.99,
	 epsilon = options.epsilon or 1e-8
      }
      optimMethod = optim.rmsprop
   elseif options.optimization == 'RPROP' then
      optimState = {
	 stepsize = options.stepsize or 0.1,
	 etaplus = options.etaplus or 1.2,
	 etaminus = options.etaminus or 0.5,
	 stepsizemax = options.stepsizemax or 50,
	 stepsizemin = options.stepsizemin or 1e-6,
	 niter = options.niter or 10
      }
      optimMethod = optim.rprop
   elseif options.optimization == 'CMAES' then
      optimState = {
	 sigma = options.sigma or 1e-3,
	 maxEval = options.maxEval or 25,
	 ftarget = options.ftarget or 1e-3,
	 popsize = options.popsize or nil
      }
      optimMethod = optim.cmaes
   else
      error('unknown optimization method')
   end
end

----------------------------------------------------------------------
-- Helper functions
----------------------------------------------------------------------

-- Pre-allocate some memory for a mini batch
local function allocate_batch(data, batchSize, options)
   local batch = torch.Tensor(data:size(options.tDim), batchSize,
			      data:size(options.featsDim))
   
   return batch
end

-- Iterate through a batch of training examples and targets using mini-batches
local function minibatchIterator(trainData, options)
   -- shuffle order at each epoch
   local shuffle = torch.randperm(trainData.data:size(options.batchDim));
   
   -- Pre-allocate mini batch space
   local inputs = allocate_batch(trainData['data'],
				 options.batchSize, options)
   local targets = allocate_batch(trainData['targets'],
				  options.batchSize, options)
   
   local t = 1
   
   return function()
      if t > trainData.data:size(options.batchDim) then
	 -- All training examples have been used, quit
	 return nil
      end
      -- disp progress
      -- xlua.progress(t, trainData.data:size(1))

      -- Check size (for last batch)
      local bSize = math.min(options.batchSize,
			     trainData.data:size(options.batchDim) - t + 1);

      -- Potential batch space memory trimming
      if (bSize ~= options.batchSize) then
	 -- Grab the opportunity to make some space
	 inputs = nil; targets = nil; collectgarbage();
	 
	 -- Pre-allocate mini batch space
	 inputs = allocate_batch(trainData['data'], bSize, options)
	 targets = allocate_batch(trainData['targets'], bSize, options)
   
	 -- Switch data to cuda
	 if options.cuda then
	    inputs = inputs:cuda();
	    targets = targets:cuda();
	 end
      end

      local k = 1;
      -- iterate over mini-batch examples
      for i = t, t+bSize-1 do
         -- select new sample
	 selectedInput = trainData.data:select(options.batchDim, shuffle[i])
	 -- store new sample
	 inputs:select(options.batchDim, k):copy(selectedInput)
         k = k + 1
      end
      
      -- Initialize targets
      if options.predict then
	 -- Train model to predict subsequent input steps
	 local k = 1;
	 for i = t, t+bSize-1 do
	    -- select new sample
	    selectedTarget = trainData.targets:select(options.batchDim,
						      shuffle[i])
	    -- store new sample
	    targets:select(options.batchDim, k):copy(selectedTarget)
	    k = k + 1
	 end
      elseif options.inpainting then
	 -- Perform bidirectional training with inpaiting criterion
	 -- TODO
	 error('TODO')
      else
	 error('Unhandled case')
      end
      
      if options.cuda then inputs = inputs:cuda(); targets = targets:cuda() end

      t = t+options.batchSize
      
      return inputs, targets
   end
end

----------------------------------------------------------------------
-- Main supervised learning function
-- TODO: harmonize with other functions (use minibatch iterator)
-- Rely on a optimization structure with options
--  . save          ('results')   - subdirectory to save/log experiments in
--  . visualize     (false)       - visualize input data and weights during training
--  . plot          (false)       - live plot
--  . optimization  ('SGD')       - optimization method: SGD | ASGD | CG | LBFGS
--  . learningRate  (1e-3)        - learning rate at t=0
--  . batchSize     (1)           - mini-batch size (1 = pure stochastic)'
--  . weightDecay   (0)           - weight decay (SGD only)
--  . momentum      (0)           - momentum (SGD only)
--  . t0            (1)           - start averaging at t0 (ASGD only), in nb of epochs
--  . maxIter       (2)           - maximum nb of iterations for CG and LBFGS
--  . type          ('float')     - type of the data: float|double|cuda
--
function supervisedTrain(model, trainData, options)
   -- epoch tracker
   epoch = epoch or 1
   -- time variable
   local time = sys.clock()

   -- Store error
   local err = 0
   
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training();
   
   -- do one epoch
   print("==> epoch # " .. epoch .. ' [batch = ' .. options.batchSize .. ']')

   -- shuffle order at each epoch
   shuffle = torch.randperm(trainData.data:size(options.batchDim));
   
   -- Pre-allocate mini batch space
   local inputs = allocate_batch(trainData, bSize, options)
   local targets = torch.zeros(options.batchSize);

   -- Switch data to cuda
   if options.cuda then
      inputs = inputs:cuda();
      targets = targets:cuda();
   end
   for t = 1,trainData.data:size(options.batchDim),options.batchSize do
      -- disp progress
      -- xlua.progress(t, trainData.data:size(1))
      -- Check size (for last batch)
      bSize = math.min(options.batchSize, trainData.data:size(options.batchDim) - t + 1);
      if (bSize ~= options.batchSize) then
	 -- Grab the opportunity to make some space
	 inputs = nil; targets = nil; collectgarbage();

	 -- Re-allocate mini batch space
	 inputs = allocate_batch(trainData, bSize, options)
   
	 -- Initialize targets
	 targets = torch.zeros(bSize);
	 
	 -- Switch data to cuda
	 if options.cuda then
	    inputs = inputs:cuda();
	    targets = targets:cuda();
	 end
      end
      local k = 1;
      -- iterate over mini-batch examples
      for i = t, t+bSize-1 do
         -- load new sample
	 inputs[k] = trainData.data[shuffle[i]];
         k = k + 1
      end
      -- Initialize targets
      -- Use prerecorded labels
      local k = 1;
      for i = t, t+bSize-1 do
	 -- load new sample
	 targets[k] = trainData.labels[shuffle[i]];
	 k = k + 1
      end
      
      local batchSize = inputs:size(1)
      local seqLen, featSize = inputs[1]:size(options.tDim), inputs[1]:size(options.featsDim)
      
      -- TODO TODO TODO --
      -- CHANGE THIS, ONLY TEMPORARY
      local sizes = inputs:size()
      local inputs_in = inputs:view(sizes[2], sizes[1], sizes[3])
      local sizes = targets:size()
      local targets_in = targets:view(sizes[2], sizes[1], sizes[3])
      
      if options.cuda then inputs_in = inputs_in:cuda() end
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
	 -- [[ Evaluate function for a complete mini-batch at once ]] --
	 -- estimate forward pass
	 
	 local output = model:forward(inputs_in)
	 -- estimate classification (compare to target)
	 -- print(output:size())
	 -- print(targets:size())
	 local err = criterion:forward(output, targets_in)
	 -- TODO
	 -- Add the sparsity here !
	 -- TODO
	 
	 -- compute overall error
	 f = f + err
	 -- estimate df/dW (perform back-prop)
	 local df_do = criterion:backward(output, targets_in)
	 model:backward(inputs_in, df_do)
	 -- in case of combined criterion
	 if (torch.type(output) == 'table') then output = output[1] end
	 
	 -- update confusion
	 for i = 1, inputs_in:size(1) do
	    confusion:add(output[i], targets_in[i])
	 end
	 
	 -- penalties (L1 and L2):
	 if options.regularizeL1 ~= 0 or options.regularizeL2 ~= 0 then
            -- locals:
            local norm,sign = torch.norm,torch.sign
            -- Loss:
            f = f + options.regularizeL1 * norm(parameters,1)
            f = f + options.regularizeL2 * norm(parameters,2) ^ 2 / 2
            -- Gradients:
            gradParameters:add(sign(parameters):mul(options.regularizeL1) + parameters:clone():mul(options.regularizeL2))
         end
	 -- return f and df/dX
	 return f, gradParameters
      end
      
      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         _,fs = optimMethod(feval, parameters, optimState)
      end
      -- TODO
      -- TODO
      -- Forget the gradient in case of recurrent model
      -- model:forget()
      -- TODO
      -- TODO
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / trainData.data:size(1)
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
   --print('==> saving model to '..filename)
   --torch.save(filename, model)
   -- next epoch
   epoch = epoch + 1
   return (1 - confusion.totalValid)
end

----------------------------------------------------------------------
-- Main reconstruction learning function
-- TODO: harmonize with other functions (use minibatch iterator)
-- Rely on a optimization structure with options
--  . save          ('results')   - subdirectory to save/log experiments in
--  . visualize     (false)       - visualize input data and weights during training
--  . plot          (false)       - live plot
--  . optimization  ('SGD')       - optimization method: SGD | ASGD | CG | LBFGS
--  . learningRate  (1e-3)        - learning rate at t=0
--  . batchSize     (1)           - mini-batch size (1 = pure stochastic)'
--  . weightDecay   (0)           - weight decay (SGD only)
--  . momentum      (0)           - momentum (SGD only)
--  . t0            (1)           - start averaging at t0 (ASGD only), in nb of epochs
--  . maxIter       (2)           - maximum nb of iterations for CG and LBFGS
--  . type          ('float')     - type of the data: float|double|cuda
--
function unsupervisedTrain(model, trainData, options)
   -- time variable
   local time = sys.clock()

   -- Store error
   local err = 0
   
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training();
   
   for inputs, targets in minibatchIterator(trainData, options) do
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
	 -- [[ Evaluate function for a complete mini-batch at once ]] --
	 -- estimate forward pass
	 
	 local output = model:forward(inputs)
	 -- estimate classification (compare to target)
	 local err = criterion:forward(output, targets)
	 -- TODO
	 -- Add the sparsity here !
	 -- TODO
	 
	 -- compute overall error
	 f = f + err
	 -- estimate df/dW (perform back-prop)
	 local gradOutputs = criterion:backward(output, targets)
	 local gradInputs = model:backward(inputs, gradOutputs)
	 
	 -- Adjust weights
	 model:updateParameters(options.learningRate)
	 
	 -- in case of combined criterion
	 if (torch.type(output) == 'table') then output = output[1] end

	 -- penalties (L1 and L2):
	 if options.regularizeL1 ~= 0 or options.regularizeL2 ~= 0 then
            -- locals:
            local norm,sign = torch.norm,torch.sign
            -- Loss:
            f = f + options.regularizeL1 * norm(parameters,1)
            f = f + options.regularizeL2 * norm(parameters,2) ^ 2 / 2
            -- Gradients:
            gradParameters:add(sign(parameters):mul(options.regularizeL1) + parameters:clone():mul(options.regularizeL2))
         end
	 -- return f and df/dX
	 return f, gradParameters
      end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         _,fs = optimMethod(feval, parameters, optimState)  -- toto was _
	 -- TODO: check the /, changed to this from a *, maybe wrong
	 local bSize = inputs:size(options.batchDim)
	 err = err + fs[1] / bSize -- so that err is indep of batch size
      end
      -- TODO
      -- TODO
      -- Forget the gradient in case of recurrent model
      -- model:forget()
      -- TODO
      -- TODO
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / trainData.data:size(options.batchDim)
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   if options.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   return err
end

----------------------------------------------------------------------
-- Supervised testing function (get forward pass error)
-- Rely on a optimization structure with options
--  . save          ('results')   - subdirectory to save/log experiments in
--  . visualize     (false)       - visualize input data and weights during training
--  . plot          (false)       - live plot
--  . optimization  ('SGD')       - optimization method: SGD | ASGD | CG | LBFGS
--  . learningRate  (1e-3)        - learning rate at t=0
--  . batchSize     (1)           - mini-batch size (1 = pure stochastic)'
--  . weightDecay   (0)           - weight decay (SGD only)
--  . momentum      (0)           - momentum (SGD only)
--  . t0            (1)           - start averaging at t0 (ASGD only), in nb of epochs
--  . maxIter       (2)           - maximum nb of iterations for CG and LBFGS
--  . type          ('float')     - type of the data: float|double|cuda
--
function supervisedTest(model, testData, options)
   -- local vars
   local time = sys.clock()
   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   -- TODO
   -- TODO
   -- Check if RNN should avoid this step ! (Seems that it will not record the time-steps in evaluation mode!)
   -- TODO
   -- TODO
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate();
   -- Pre-allocate mini batch space
   local inputs = {};
   if (testData.data[1]:nDimension() == 1) then
      inputs = torch.Tensor(options.batchSize, testData.data[1]:size(1))
   else
      inputs = torch.Tensor(options.batchSize, testData.data[1]:size(1), testData.data[1]:size(2))
   end
   local targets = torch.zeros(options.batchSize);
   -- Switch data to cuda
   if options.cuda then
      inputs = inputs:cuda();
      targets = targets:cuda();
   end
   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData.data:size(1),options.batchSize do
      -- disp progress
      --xlua.progress(t, testData.data:size(1))
      -- Check size of batch (for last smaller)
      bSize = math.min(options.batchSize, testData.data:size(1) - t + 1);
      if (bSize ~= options.batchSize) then
	 if (testData.data[1]:nDimension() == 1) then
	    inputs = torch.Tensor(bSize, testData.data[1]:size(1))
	 else
	    inputs = torch.Tensor(bSize, testData.data[1]:size(1), testData.data[1]:size(2))
	 end
	 targets = torch.zeros(bSize);
	 -- Switch data to cuda
	 if options.cuda then
	    inputs = inputs:cuda();
	    targets = targets:cuda();
	 end
      end
      -- iterate over mini-batch examples
      local k = 1;
      for i = t,math.min(t+options.batchSize-1,testData.data:size(1)) do
	 inputs[k] = testData.data[i];
	 targets[k] = testData.labels[i];
	 k = k + 1;
      end
      -- test sample
      local pred = model:forward(inputs)
      -- in case of combined criterion
      if (torch.type(pred) == 'table') then pred = pred[1]; end
      for i = 1,k-1 do
	 confusion:add(pred[i], targets[i])
      end
   end
   -- timing
   time = sys.clock() - time
   time = time / testData.data:size(1)
   --print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
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

function unsupervisedTest(model, testData, options)
   -- local vars
   local time = sys.clock()
   local err = math.huge

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   -- TODO
   -- TODO
   -- Check if RNN should avoid this step ! (Seems that it will not record the time-steps in evaluation mode!)
   -- TODO
   -- TODO
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate();

   for inputs, targets in minibatchIterator(testData, options) do
      -- test sample
      local pred = model:forward(inputs)
      -- in case of combined criterion
      if (torch.type(pred) == 'table') then pred = pred[1]; end

      err = criterion:forward(pred, targets)
   end
   -- timing
   time = sys.clock() - time
   time = time / testData.data:size(options.batchDim)
   --print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   -- next iteration:
   -- confusion:zero()
   return err
end

----------------------------------------------------------------------
-- Unsupervised learning function with tables
-- Mainly used for recurrent networks
----------------------------------------------------------------------
function unsupervisedTable(model, testData, params)
   -- are we using the hessian?
   if params.hessian then
      model:initDiagHessianParameters()
   end
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training();
   -- get all parameters
   x,dl_dx,ddl_ddx = model:getParameters();
   -- training errors
   local err = 0
   local iter = 0
   -- create mini batch
   local bSize = params.batchSize
   local inputs = {};
   local targets = {};
   if (testData.data[1]:nDimension() == 2) then
      for i = 1,#testData.data do
	 inputs[i] = torch.Tensor(bSize, testData.data[1]:size(2))
	 targets[i] = torch.Tensor(bSize, testData.data[1]:size(2))
	 if options.cuda then inputs[i]:cuda(); targets[i]:cuda() end
      end
   else
      for i = 1,#testData.data do
	 inputs[i] = torch.Tensor(bSize, testData.data[1]:size(2), testData.data[1]:size(3))
	 targets[i] = torch.Tensor(bSize, testData.data[1]:size(2), testData.data[1]:size(3))
	 if options.cuda then inputs[i]:cuda(); targets[i]:cuda(); end
      end
   end
   for t = 1, math.min(params.maxIter, (testData.data[1]:size(1)-params.batchSize)),params.batchSize do
      -- progress
      iter = iter+1
      --xlua.progress(iter*params.batchSize, testData.data[1]:size(1))
      -- Check size of batch (for last smaller)
      bSize = math.min(options.batchSize, testData.data[1]:size(1) - t + 1)
      -- iterate over mini-batch examples
      for k = 1,#testData.data do
	 for i = t,math.min(t+options.batchSize-1,testData.data:size(1)) do
	    inputs[k] = testData.data[k][i];
	    targets[k] = testData.data[k][i];
	    --
	    -- TODO
	    -- This is where to add noise, warp, outlier, etc ...
	    -- Or should I do this inside the construction of the unsupervised dataset ?
	    -- TODO
	    --
	    -- COMMENTED THIS OUT, useless in a for loop
	    -- k = k + 1;
	 end
	 -- optimize on current mini-batch
	 _,fs = optimMethod(feval, x, optimState)
	 err = err + fs[1] * params.batchSize -- so that err is indep of batch size
	 -- TODO
	 -- Reset the model gradients in case of recurrent model
	 -- model:forget()
	 -- TODO
      end
      return err;
   end
end
----------------------------------------------------------------------
-- Main unsupervised learning function
-- Rely on a optimization structure with options
--  . save          ('results')   - subdirectory to save/log experiments in
--  . visualize     (false)       - visualize input data and weights during training
--  . plot          (false)       - live plot
--  . optimization  ('SGD')       - optimization method: SGD | ASGD | CG | LBFGS
--  . learningRate  (1e-3)        - learning rate at t=0
--  . batchSize     (1)           - mini-batch size (1 = pure stochastic)'
--  . weightDecay   (0)           - weight decay (SGD only)
--  . momentum      (0)           - momentum (SGD only)
--  . t0            (1)           - start averaging at t0 (ASGD only), in nb of epochs
--  . maxIter       (2)           - maximum nb of iterations for CG and LBFGS
--  . type          ('float')     - type of the data: float|double|cuda
--
function __unsupervisedTrain_old(model, testData, params)
   -- check if we are working with a table
   if torch.type(testData.data) == 'table' then
      return unsupervisedTable(model, testData, params);
   end
   -- are we using the hessian?
   if params.hessian then
      model:initDiagHessianParameters()
   end
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training();
   -- get all parameters
   x,dl_dx,ddl_ddx = model:getParameters();
   -- training errors
   local err = 0
   local iter = 0
   -- create mini batch
   local inputs = {};
   local targets = {};
   if (testData.data[1]:nDimension() == 1) then
      inputs = torch.Tensor(options.batchSize, testData.data[1]:size(1))
      targets = torch.Tensor(options.batchSize, testData.data[1]:size(1))
   else
      inputs = torch.Tensor(options.batchSize, testData.data[1]:size(1), testData.data[1]:size(2))
      targets = torch.Tensor(options.batchSize, testData.data[1]:size(1), testData.data[1]:size(2))
   end
   if options.cuda then inputs = inputs:cuda(); targets = targets:cuda(); end
   for t = 1,math.min(params.maxIter, (testData.data:size(1)-params.batchSize)),params.batchSize do
      -- update diagonal hessian parameters
      if params.hessian and math.fmod(t , params.hessianinterval) == 1 then
	 -- some extra vars:
	 local hessiansamples = params.hessiansamples
	 local minhessian = params.minhessian
	 local maxhessian = params.maxhessian
	 local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
	 etas = etas or ddl_ddx:clone()
	 for i = 1,hessiansamples do
	    -- next
	    local ex = testData.data[i];
	    if options.cuda then ex:cuda(); end
	    local input = ex;
	    local target = ex;
	    model:updateOutput(input, target)
	    -- gradient
	    dl_dx:zero()
	    model:updateGradInput(input, target)
	    model:accGradParameters(input, target)
	    -- hessian
	    ddl_ddx:zero()
	    model:updateDiagHessianInput(input, target)
	    model:accDiagHessianParameters(input, target)
	    -- accumulate
	    ddl_ddx_avg:add(1/hessiansamples, ddl_ddx)
	 end
	 -- cap hessian params
	 print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
	 ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
	 ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
	 print('==> corrected ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
	 -- generate learning rates
	 etas:fill(1):cdiv(ddl_ddx_avg)
      end
      -- progress
      iter = iter+1
      --xlua.progress(iter*params.batchSize, testData.data:size(1));
      -- iterate over mini-batch examples
      -- Check size of batch (for last smaller)
      local bSize = math.min(options.batchSize, testData.data:size(1) - t + 1);
      local k = 1;
      for i = t,math.min(t+options.batchSize-1,testData.data:size(1)) do
	 inputs[k] = testData.data[i];
	 targets[k] = testData.data[i];
	 --if options.cuda then inputs[k] = inputs[k]:cuda(); targets[k] = targets[k]:cuda(); end
	 
	 --
	 -- TODO
	 -- This is where to add noise, warp, outlier, etc ...
	 -- Or should I do this inside the construction of the unsupervised dataset ?
	 -- TODO
	 --
	 
	 k = k + 1;
      end
      -- define eval closure
      local feval = function()
	 -- reset gradient/f
	 local f = 0
	 --model:forget()
	 dl_dx:zero()
	 --
	 -- TODO FOR ALL TRAINING METHODS !
	 -- GRADIENT CLIPPING IN CASE OF RECURRENT MODEL !
	 -- if opt.cutoffNorm > 0 then
	 --   local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
	 --         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
	 --   
	 -- model:maxParamNorm(opt.maxOutNorm) -- affects params 
	 --
	 --
	 --
	 -- f
	 f = f + model:updateOutput(inputs, targets)
	 --f = f+model:forward(inputs,targets);
	 -- gradients
	 model:updateGradInput(inputs, targets)
	 model:accGradParameters(inputs, targets)
	 -- normalize
	 -- dl_dx:div(#inputs)
	 -- f = f/#inputs
	 -- return f and df/dx
	 return f,dl_dx
      end
      -- optimize on current mini-batch
      _,fs = optimMethod(feval, x, optimState)
      local bSize = inputs:size(2)
      err = err + fs[1] * bSize -- so that err is indep of batch size
      -- normalize
      if params.model:find('psd') then
	 model:normalize()
      end
      -- TODO
      -- Reset the model gradients in case of recurrent model
      -- model:forget();
      -- TODO
   end
   return err;
end

----------------------------------------------------------------------
-- Unsupervised testing for table
----------------------------------------------------------------------
function unsupervisedTestTable(model, testData, params)
   -- training errors
   local err = 0
   local iter = 0
   local time = sys.clock();
   -- Switch model to evaluate mode
   model:evaluate();
   -- Update the error of the model
   err = err + model:updateOutput(testData.data, testData.data)
   -- timing
   time = sys.clock() - time
   time = time / testData.data[1]:size(1)
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   err = err / testData.data[1]:size(1);
   return err;
end

----------------------------------------------------------------------
-- Main unsupervised learning function
-- Rely on a optimization structure with options
--  . save          ('results')   - subdirectory to save/log experiments in
--  . visualize     (false)       - visualize input data and weights during training
--  . plot          (false)       - live plot
--  . optimization  ('SGD')       - optimization method: SGD | ASGD | CG | LBFGS
--  . learningRate  (1e-3)        - learning rate at t=0
--  . batchSize     (1)           - mini-batch size (1 = pure stochastic)'
--  . weightDecay   (0)           - weight decay (SGD only)
--  . momentum      (0)           - momentum (SGD only)
--  . t0            (1)           - start averaging at t0 (ASGD only), in nb of epochs
--  . maxIter       (2)           - maximum nb of iterations for CG and LBFGS
--  . type          ('float')     - type of the data: float|double|cuda
--
function __unsupervisedTest_old(model, testData, params)
   -- check if we are working with a table
   if torch.type(testData.data) == 'table' then
      return unsupervisedTestTable(model, testData, params);
   end
   -- training errors
   local err = 0
   local iter = 0
   local time = sys.clock();
   -- Switch model to evaluate mode
   model:evaluate();
   if options.cuda then testDataTmp = testData.data:cuda(); end
   -- PRIOR TO THAT I WAS CLONING BOTH OF THEM
   err = err + model:updateOutput(testDataTmp, testDataTmp)
   --for i = 1,testData.data:size(1) do
   -- progress
   --  iter = iter+1
   --  xlua.progress(iter*params.batchSize, params.statinterval)
   -- create mini-batch
   --  local example = testData.data[t]
   -- load new sample
   --  local sample = testData.data[i]
   --  if options.cuda then sample:cuda(); end
   --  local input = sample:clone()
   --  local target = sample:clone()
   --  err = err + model:forward(input, target)
   --end
   -- timing
   time = sys.clock() - time
   time = time / testData.data:size(1)
   --print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   err = err / testData.data:size(1);
   return err;
end

--
-- TODO
-- Need to adapt this code as a SUPERVISED Hessian-free
-- In which case, I also need to embed the criterion inside
-- [[ LATERS ]]
-- TODO
--
----------------------------------------------------------------------
-- Supervised learning function with Hessian-Free mechanism.
-- Rely on a optimization structure with options
--  . save          ('results')   - subdirectory to save/log experiments in
--  . visualize     (false)       - visualize input data and weights during training
--  . plot          (false)       - live plot
--  . optimization  ('SGD')       - optimization method: SGD | ASGD | CG | LBFGS
--  . learningRate  (1e-3)        - learning rate at t=0
--  . batchSize     (1)           - mini-batch size (1 = pure stochastic)'
--  . weightDecay   (0)           - weight decay (SGD only)
--  . momentum      (0)           - momentum (SGD only)
--  . t0            (1)           - start averaging at t0 (ASGD only), in nb of epochs
--  . maxIter       (2)           - maximum nb of iterations for CG and LBFGS
--  . type          ('float')     - type of the data: float|double|cuda
--
function supervisedTrainHF(model, testData, params)
   -- are we using the hessian?
   if params.hessian then
      model:initDiagHessianParameters()
   end
   -- get all parameters
   x,dl_dx,ddl_ddx = model:getParameters()
   -- training errors
   local err = 0
   local iter = 0
   for t = 1,params.maxIter,params.batchSize do
      -- update diagonal hessian parameters
      if params.hessian and math.fmod(t , params.hessianinterval) == 1 then
	 -- some extra vars:
	 local hessiansamples = params.hessiansamples
	 local minhessian = params.minhessian
	 local maxhessian = params.maxhessian
	 local ddl_ddx_avg = ddl_ddx:clone(ddl_ddx):zero()
	 etas = etas or ddl_ddx:clone()
	 for i = 1,hessiansamples do
	    -- next
	    local ex = testData.data[i];
	    if options.cuda then ex:cuda(); end
	    local input = ex;
	    local target = ex;
	    model:updateOutput(input, target)
	    -- gradient
	    dl_dx:zero()
	    model:updateGradInput(input, target)
	    model:accGradParameters(input, target)
	    -- hessian
	    ddl_ddx:zero()
	    model:updateDiagHessianInput(input, target)
	    model:accDiagHessianParameters(input, target)
	    -- accumulate
	    ddl_ddx_avg:add(1/hessiansamples, ddl_ddx)
	 end
	 -- cap hessian params
	 print('==> ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
	 ddl_ddx_avg[torch.lt(ddl_ddx_avg,minhessian)] = minhessian
	 ddl_ddx_avg[torch.gt(ddl_ddx_avg,maxhessian)] = maxhessian
	 print('==> corrected ddl/ddx : min/max = ' .. ddl_ddx_avg:min() .. '/' .. ddl_ddx_avg:max())
	 -- generate learning rates
	 etas:fill(1):cdiv(ddl_ddx_avg)
      end
      -- progress
      iter = iter+1
      xlua.progress(iter*params.batchSize, params.statinterval)
      -- create mini-batch
      local example = testData.data[t]
      local inputs = {}
      local targets = {}
      for i = t,t+params.batchSize-1 do
	 -- load new sample
	 local sample = testData.data[i]
	 if options.cuda then sample:cuda(); end
	 local input = sample:clone()
	 local target = sample:clone()
	 table.insert(inputs, input)
	 table.insert(targets, target)
      end
      -- define eval closure
      local feval = function()
	 -- reset gradient/f
	 local f = 0
	 dl_dx:zero()
	 -- estimate f and gradients, for minibatch
	 for i = 1,#inputs do
	    -- f
	    f = f + model:updateOutput(inputs[i], targets[i])
	    -- gradients
	    model:updateGradInput(inputs[i], targets[i])
	    model:accGradParameters(inputs[i], targets[i])
	 end
	 -- normalize
	 dl_dx:div(#inputs)
	 f = f/#inputs
	 -- return f and df/dx
	 return f,dl_dx
      end
      -- optimize on current mini-batch
      _,fs = optimMethod(feval, x, params)
      err = err + fs[1] * params.batchSize -- so that err is indep of batch size
      -- normalize
      if params.model:find('psd') then
	 model:normalize()
      end
   end
end
