----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- The optime package contains several optimization routines for Torch. Most optimization algorithms has the following interface:
--
-- x*, {f}, ... = optim.method(opfunc, x, state)
-- 
-- opfunc : a user-defined closure that respects this API: f, df/dx = func(x)
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
-- Main supervised learning function
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
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training();
   -- shuffle order at each epoch
   shuffle = torch.randperm(trainData.data:size(1));
   -- do one epoch
   print("==> epoch # " .. epoch .. ' [batch = ' .. options.batchSize .. ']')
   for t = 1,trainData.data:size(1),options.batchSize do
      -- disp progress
      xlua.progress(t, trainData.data:size(1))
      -- create mini batch
      local inputs = {}
      local targets = {}
      -- iterate over mini-batch examples
      for i = t,math.min(t+options.batchSize-1,trainData.data:size(1)) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         if options.type == 'double' then input = input:double() end
         if options.cuda then input = input:cuda() end
         table.insert(inputs, input)
         table.insert(targets, target)
      end
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
        -- evaluate function for each example in the mini batch
        for i = 1,#inputs do
          -- estimate forward pass
          local output = model:forward(inputs[i])
          -- estimate classification (compare to target)
          local err = criterion:forward(output, targets[i])
          -- TODO
          -- Add the sparsity here !
          -- TODO
          -- compute overall error
          f = f + err
          -- estimate df/dW (perform back-prop)
          local df_do = criterion:backward(output, targets[i])
          model:backward(inputs[i], df_do)
          -- update confusion
          confusion:add(output, targets[i])
        end
        -- normalize gradients and f(X)
        gradParameters:div(#inputs)
        f = f/#inputs
        -- return f and df/dX
        return f,gradParameters
      end
      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
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
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
   -- next epoch
   epoch = epoch + 1
   return (1 - confusion.totalValid);
end

----------------------------------------------------------------------
-- Main supervised learning function
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
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate();
   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData.data:size(1) do
      -- disp progress
      xlua.progress(t, testData.data:size(1))
      -- get new sample
      local input = testData.data[t]
      if options.type == 'double' then input = input:double() end
      if options.cuda then input = input:cuda() end
      local target = testData.labels[t]
      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
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
function unsupervisedTrain(model, testData, params)
  -- are we using the hessian?
  if params.hessian then
    model:initDiagHessianParameters()
  end
  -- get all parameters
  x,dl_dx,ddl_ddx = model:getParameters();
  -- training errors
  local err = 0
  local iter = 0
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
        --
        -- TODO
        -- This is where to add noise, warp, outlier, etc ...
        -- Or should I do this inside the construction of the unsupervised dataset ?
        -- TODO
        --
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
    for i = t,math.min(t+params.batchSize-1,testData.data:size(1)) do
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
    _,fs = optimMethod(feval, x, optimState)
    err = err + fs[1] * params.batchSize -- so that err is indep of batch size
    -- normalize
    if params.model:find('psd') then
      model:normalize()
    end
  end
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
  epoch = epoch + 1;
end