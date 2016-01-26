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

modelGAN = {};

----------------------------------------------------------------------
-- Handmade Generalized Adversarial Network
----------------------------------------------------------------------

function modelGAN.defineGAN()
 ----------------------------------------------------------------------
  -- define D network to train
  model_D = nn.Sequential()
  model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(cudnn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2))
  model_D:add(cudnn.ReLU(true))
  model_D:add(cudnn.SpatialMaxPooling(2,2))
  model_D:add(nn.SpatialDropout(0.2))
  model_D:add(nn.Reshape(8*8*96))
  model_D:add(nn.Linear(8*8*96, 1024))
  model_D:add(cudnn.ReLU(true))
  model_D:add(nn.Dropout())
  model_D:add(nn.Linear(1024,1))
  model_D:add(nn.Sigmoid())

  x_input = nn.Identity()()
  lg = nn.Linear(opt.noiseDim, 128*8*8)(x_input)
  lg = nn.Reshape(128, 8, 8)(lg)
  lg = cudnn.ReLU(true)(lg)
  lg = nn.SpatialUpSamplingNearest(2)(lg)
  lg = cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, 2, 2)(lg)
  lg = nn.SpatialBatchNormalization(256)(lg)
  lg = cudnn.ReLU(true)(lg)
  lg = nn.SpatialUpSamplingNearest(2)(lg)
  lg = cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2)(lg)
  lg = nn.SpatialBatchNormalization(256)(lg)
  lg = cudnn.ReLU(true)(lg)
  lg = nn.SpatialUpSamplingNearest(2)(lg)
  lg = cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2)(lg)
  lg = nn.SpatialBatchNormalization(128)(lg)
  lg = cudnn.ReLU(true)(lg)
  lg = cudnn.SpatialConvolution(128, 3, 3, 3, 1, 1, 1, 1)(lg)
  model_G = nn.gModule({x_input}, {lg})
  
  -- loss function: negative log-likelihood
  criterion = nn.BCECriterion()

  -- retrieve parameters and gradients
  parameters_D,gradParameters_D = model_D:getParameters()
  parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

  return model_D, model_G;
end

--[[

OTHER WAYS OF DEFINING THE NETWORK :
if opt.network ~= '' then
  print('<trainer> reloading previously trained network: ' .. opt.network)
  local tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
  print('Discriminator network:')
  print(model_D)
  print('Generator network:')
  print(model_G)
elseif opt.model == 'small' then
   local nplanes = 64
   model_D = nn.Sequential()
   model_D:add(nn.CAddTable())
   model_D:add(nn.SpatialConvolution(3, nplanes, 5, 5)) --28 x 28
   model_D:add(nn.ReLU())
   model_D:add(nn.SpatialConvolution(nplanes, nplanes, 5, 5, 2, 2))
   local sz = math.floor( ( (opt.fineSize - 5 + 1) - 5) / 2 + 1)
   model_D:add(nn.View(nplanes*sz*sz))
   model_D:add(nn.ReLU())
   model_D:add(nn.Linear(nplanes*sz*sz, 1))
   model_D:add(nn.Sigmoid())
   local nplanes = 128
   model_G = nn.Sequential()
   model_G:add(nn.JoinTable(2, 2))
   model_G:add(cudnn.SpatialConvolutionUpsample(3+1, nplanes, 7, 7, 1))
   model_G:add(nn.ReLU())
   model_G:add(cudnn.SpatialConvolutionUpsample(nplanes, nplanes, 7, 7, 1))
   model_G:add(nn.ReLU())
   model_G:add(cudnn.SpatialConvolutionUpsample(nplanes, 3, 5, 5, 1))
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
elseif opt.model == 'large' then
   require 'fbcunn'
   print('Generator network (good):')
   desc_G = '___JT22___C_4_64_g1_7x7___R__BN___C_64_368_g4_7x7___R__BN___SDrop 0.5___C_368_128_g4_7x7___R__BN___P_LPOut_2___C_64_224_g2_5x5___R__BN___SDrop 0.5___C_224_3_g1_7x7__BNA'
   model_G = nn.Sequential()
   model_G:add(nn.JoinTable(2, 2))
   model_G:add(cudnn.SpatialConvolutionUpsample(3+1, 64, 7, 7, 1, 1)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(64, nil, nil, false))
   model_G:add(cudnn.SpatialConvolutionUpsample(64, 368, 7, 7, 1, 4)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(368, nil, nil, false))
   model_G:add(nn.SpatialDropout(0.5))
   model_G:add(cudnn.SpatialConvolutionUpsample(368, 128, 7, 7, 1, 4)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(128, nil, nil, false))
   model_G:add(nn.FeatureLPPooling(2,2,2,true))
   model_G:add(cudnn.SpatialConvolutionUpsample(64, 224, 5, 5, 1, 2)):add(cudnn.ReLU(true))
   model_G:add(nn.SpatialBatchNormalization(224, nil, nil, false))
   model_G:add(nn.SpatialDropout(0.5))
   model_G:add(cudnn.SpatialConvolutionUpsample(224, 3, 7, 7, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(3, nil, nil, false))
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   print(desc_G)

   desc_D = '___CAdd___C_3_48_g1_3x3___R___C_48_448_g4_5x5___R___C_448_416_g16_7x7___R___V_166400___L 166400_1___Sig'
   model_D = nn.Sequential()
   model_D:add(nn.CAddTable())
   model_D:add(cudnn.SpatialConvolution(3, 48, 3, 3))
   model_D:add(cudnn.ReLU(true))
   model_D:add(cudnn.SpatialConvolution(48, 448, 5, 5, 1, 1, 0, 0, 4))
   model_D:add(cudnn.ReLU(true))
   model_D:add(cudnn.SpatialConvolution(448, 416, 7, 7, 1, 1, 0, 0, 16))
   model_D:add(cudnn.ReLU())
   model_D:cuda()
   local dummy_input = torch.zeros(opt.batchSize, 3, opt.fineSize, opt.fineSize):cuda()
   local out = model_D:forward({dummy_input, dummy_input})
   local nElem = out:nElement() / opt.batchSize
   model_D:add(nn.View(nElem):setNumInputDims(3))
   model_D:add(nn.Linear(nElem, 1))
   model_D:add(nn.Sigmoid())
   model_D:cuda()
   print(desc_D)
elseif opt.model == 'autogen' then
   -- define G network to train
   print('Generator network:')
   model_G,desc_G = generateModelG(3,5,128,512,3,7, 'mixed', 0, 4, 2, true)
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   print(desc_G)
   print(model_G)
   local trygen = 1
   local poolType = 'none'
   -- if torch.random(1,2) == 1 then poolType = 'none' end
   repeat
      trygen = trygen + 1
      if trygen == 500 then error('Could not find a good D model') end
      -- define D network to train
      print('Discriminator network:')
      model_D,desc_D = generateModelD(2,6,64,512,3,7, poolType, 0, 4, 2)
      print(desc_D)
      print(model_D)
   until (freeParams(model_D) < freeParams(model_G))
      and (freeParams(model_D) > freeParams(model_G) / 10)
elseif opt.model == 'full' or opt.model == 'fullgen' then
   local nhid = 1024
   local nhidlayers = 2
   local batchnorm = 1 -- disabled
   if opt.model == 'fullgen' then
      nhidlayers = torch.random(1,5)
      nhid = torch.random(8, 128) * 16
      batchnorm = torch.random(1,2)
   end
   desc_G = ''
   model_G = nn.Sequential()
   model_G:add(nn.JoinTable(2, 2))
   desc_G = desc_G .. '___JT22'
   model_G:add(nn.View(4 * opt.fineSize * opt.fineSize):setNumInputDims(3))
   desc_G = desc_G .. '___V_' .. 4 * opt.fineSize * opt.fineSize
   model_G:add(nn.Linear(4 * opt.fineSize * opt.fineSize, nhid)):add(nn.ReLU())
   desc_G = desc_G .. '___L ' .. 4 * opt.fineSize * opt.fineSize .. '_' .. nhid
   desc_G = desc_G .. '__R'
   if batchnorm == 2 then
      model_G:add(nn.BatchNormalization(nhid), nil, nil, true)
      desc_G = desc_G .. '__BNA'
   end
   model_G:add(nn.Dropout(0.5))
   desc_G = desc_G .. '__Drop' .. 0.5
   for i=1,nhidlayers do
      model_G:add(nn.Linear(nhid, nhid)):add(nn.ReLU())
      desc_G = desc_G .. '___L ' .. nhid .. '_' .. nhid
      desc_G = desc_G .. '__R'
      if batchnorm == 2 then
         model_G:add(nn.BatchNormalization(nhid), nil, nil, true)
         desc_G = desc_G .. '__BNA'
      end
      model_G:add(nn.Dropout(0.5))
      desc_G = desc_G .. '__Drop' .. 0.5
   end
   model_G:add(nn.Linear(nhid, opt.geometry[1]*opt.geometry[2]*opt.geometry[3]))
   desc_G = desc_G .. '___L ' .. nhid .. '_' .. opt.geometry[1]*opt.geometry[2]*opt.geometry[3]
   if batchnorm == 2 then
      model_G:add(nn.BatchNormalization(opt.geometry[1]*opt.geometry[2]*opt.geometry[3]))
      desc_G = desc_G .. '__BNA'
   end
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
   desc_G = desc_G .. '___V_' .. opt.geometry[1] .. '_' ..  opt.geometry[2] .. '_' ..  opt.geometry[3]
   print(desc_G)
   print(model_G)

   nhid = nhid / 2
   desc_D = ''
   model_D = nn.Sequential()
   model_D:add(nn.CAddTable())
   desc_D = desc_D .. '___CAdd'
   model_D:add(nn.View(opt.geometry[1]* opt.geometry[2]* opt.geometry[3]))
   desc_D = desc_D .. '___V_' .. opt.geometry[1]* opt.geometry[2]* opt.geometry[3]
   model_D:add(nn.Linear(opt.geometry[1]* opt.geometry[2]* opt.geometry[3], nhid)):add(nn.ReLU())
   desc_D = desc_D .. '___L ' .. opt.geometry[1]* opt.geometry[2]* opt.geometry[3] .. '_' .. nhid
   desc_D = desc_D .. '__R'
   for i=1,nhidlayers do
      model_D:add(nn.Linear(nhid, nhid)):add(nn.ReLU())
      desc_D = desc_D .. '___L ' .. nhid .. '_' .. nhid
      desc_D = desc_D .. '__R'
      model_D:add(nn.Dropout(0.5))
      desc_D = desc_D .. '__Drop' .. 0.5
   end
   model_D:add(nn.Linear(nhid, 1))
   desc_D = desc_D .. '___L ' .. nhid .. '_' .. 1
   model_D:add(nn.Sigmoid())
   desc_D = desc_D .. '__Sig'
   model_D:cuda()
   print(desc_D)
   print(model_D)
elseif opt.model == 'small_18' then
   assert(opt.scratch == 1) -- check that this is not conditional on a previous scale
   ----------------------------------------------------------------------
   local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]
   -- define D network to train
   local numhid = 600
   model_D = nn.Sequential()
   model_D:add(nn.View(input_sz):setNumInputDims(3))
   model_D:add(nn.Linear(input_sz, numhid))
   model_D:add(nn.ReLU())
   model_D:add(nn.Dropout())
   model_D:add(nn.Linear(numhid, numhid))
   model_D:add(nn.ReLU())
   model_D:add(nn.Dropout())
   model_D:add(nn.Linear(numhid, 1))
   model_D:add(nn.Sigmoid())
   ----------------------------------------------------------------------
   local noiseDim = opt.noiseDim[1] * opt.noiseDim[2] * opt.noiseDim[3]
   -- define G network to train
   local numhid = 600
   model_G = nn.Sequential()
   model_G:add(nn.View(noiseDim):setNumInputDims(3))
   model_G:add(nn.Linear(noiseDim, numhid))
   model_G:add(nn.ReLU())
   model_G:add(nn.Linear(numhid, numhid))
   model_G:add(nn.ReLU())
   model_G:add(nn.Linear(numhid, input_sz))
   model_G:add(nn.View(opt.geometry[1], opt.geometry[2], opt.geometry[3]))
end

-- training function
function adversarial.train(dataset, N)
  model_G:training()
  model_D:training()
  epoch = epoch or 1
  local N = N or dataset:size()[1]
  local dataBatchSize = opt.batchSize / 2
  local time = sys.clock()

  -- do one epoch
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')
  for t = 1,N,dataBatchSize do

    local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local targets = torch.Tensor(opt.batchSize)
    local noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      --  forward pass
      local outputs = model_D:forward(inputs)
     -- err_F = criterion:forward(outputs:narrow(1, 1, opt.batchSize / 2), targets:narrow(1, 1, opt.batchSize / 2))
     -- err_R = criterion:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2), targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2))
     err_R = criterion:forward(outputs:narrow(1, 1, opt.batchSize / 2), targets:narrow(1, 1, opt.batchSize / 2))
     err_F = criterion:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2), targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2))
    
    local margin = 0.3
      sgdState_D.optimize = true
      sgdState_G.optimize = true      
      if err_F < margin or err_R < margin then
         sgdState_D.optimize = false
      end
      if err_F > (1.0-margin) or err_R > (1.0-margin) then
         sgdState_G.optimize = false
      end
      if sgdState_G.optimize == false and sgdState_D.optimize == false then
         sgdState_G.optimize = true 
         sgdState_D.optimize = true
      end

  
      --print(monA:size(), tarA:size())
      io.write("v1_lfw| R:", err_R,"  F:", err_F, "  ")
      local f = criterion:forward(outputs, targets)

      -- backward pass 
      local df_do = criterion:backward(outputs, targets)
      model_D:backward(inputs, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end
      -- update confusion (add 1 since targets are binary)
      for i = 1,opt.batchSize do
        local c
        if outputs[i][1] > 0.5 then c = 2 else c = 1 end
        confusion:add(c, targets[i]+1)
      end
      --print('grad D', gradParameters_D:norm())
      return f,gradParameters_D
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      if x ~= parameters_G then -- get new parameters
        parameters_G:copy(x)
      end
      
      gradParameters_G:zero() -- reset gradients

      -- forward pass
      local samples = model_G:forward(noise_inputs)
      local outputs = model_D:forward(samples)
      local f = criterion:forward(outputs, targets)
     io.write("G:",f, " G:", tostring(sgdState_G.optimize)," D:",tostring(sgdState_D.optimize)," ", sgdState_G.numUpdates, " ", sgdState_D.numUpdates , "\n")
      io.flush()

      --  backward pass
      local df_samples = criterion:backward(outputs, targets)
      model_D:backward(samples, df_samples)
      local df_do = model_D.modules[1].gradInput
      model_G:backward(noise_inputs, df_do)
      print('gradParameters_G', gradParameters_G:norm())
      return f,gradParameters_G
    end

    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    -- Get half a minibatch of real, half fake
    for k=1,opt.K do
      -- (1.1) Real data 
      local k = 1
      for i = t,math.min(t+dataBatchSize-1,dataset:size()[1]) do
        local idx = math.random(dataset:size()[1])
        local sample = dataset[idx]
        inputs[k] = sample:clone()
        k = k + 1
      end
      targets[{{1,dataBatchSize}}]:fill(1)
      -- (1.2) Sampled data
      noise_inputs:normal(0, 1)
      local samples = model_G:forward(noise_inputs[{{dataBatchSize+1,opt.batchSize}}])
      for i = 1, dataBatchSize do
        inputs[k] = samples[i]:clone()
        k = k + 1
      end
      targets[{{dataBatchSize+1,opt.batchSize}}]:fill(0)

      rmsprop(fevalD, parameters_D, sgdState_D)

    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:normal(0, 1)
    targets:fill(1)
    rmsprop(fevalG, parameters_G, sgdState_G)

    -- display progress
    xlua.progress(t, dataset:size()[1])
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()[1]
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()

  -- save/log current net
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, 'adversarial.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D = model_D, G = model_G, opt = opt})
  end

  -- next epoch
  epoch = epoch + 1
end

  sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
  sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
  sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
  sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)

DOUBLE CONDITIONAL TRAINING ?

require 'torch'
require 'optim'
require 'pl'
require 'paths'
require 'image'

local adversarial = {}

-- training function
function adversarial.train(dataset, N)
  epoch = epoch or 1
  local N = N or dataset:size()
  local time = sys.clock()
  local dataBatchSize = opt.batchSize / 2

  local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local targets = torch.Tensor(opt.batchSize)
  local noise_inputs 
  if type(opt.noiseDim) == 'number' then
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)
  else
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  end
  local cond_inputs1
  local cond_inputs2
  if type(opt.condDim1) == 'number' then
    cond_inputs1 = torch.Tensor(opt.batchSize, opt.condDim1)
  else
    cond_inputs1 = torch.Tensor(opt.batchSize, opt.condDim1[1], opt.condDim1[2], opt.condDim1[3])
  end
  if type(opt.condDim2) == 'number' then
    cond_inputs2 = torch.Tensor(opt.batchSize, opt.condDim2)
  else
    cond_inputs2 = torch.Tensor(opt.batchSize, opt.condDim2[1], opt.condDim2[2], opt.condDim2[3])
  end

  -- do one epoch
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ', momentum = ' .. sgdState_D.momentum .. ']')
  for t = 1,N,dataBatchSize*opt.K do 

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      --  forward pass
      local outputs = model_D:forward({inputs, cond_inputs1, cond_inputs2})
      local f = criterion:forward(outputs, targets)

      -- backward pass 
      local df_do = criterion:backward(outputs, targets)
      model_D:backward({inputs, cond_inputs1, cond_inputs2}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end
      -- update confusion (add 1 since classes are binary)
      for i = 1,opt.batchSize do
        local c
        if outputs[i][1] > 0.5 then c = 2 else c = 1 end
        confusion:add(c, targets[i]+1)
      end

      return f,gradParameters_D
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      if x ~= parameters_G then -- get new parameters
        parameters_G:copy(x)
      end
      
      gradParameters_G:zero() -- reset gradients

      -- forward pass
      local samples = model_G:forward({noise_inputs, cond_inputs1, cond_inputs2})
      local outputs = model_D:forward({samples, cond_inputs1, cond_inputs2})
      local f = criterion:forward(outputs, targets)

      --  backward pass
      local df_samples = criterion:backward(outputs, targets)
      model_D:backward({samples, cond_inputs1, cond_inputs2}, df_samples)
      local df_do = model_D.gradInput[1]
      model_G:backward({noise_inputs, cond_inputs1, cond_inputs2}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_G:add( sign(parameters_G):mul(opt.coefL1) + parameters_G:clone():mul(opt.coefL2) )
      end

      return f,gradParameters_G
    end

    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    -- Get half a minibatch of real, half fake
    for k=1,opt.K do
      -- (1.1) Real data 
      local k = 1
      for i = t,math.min(t+dataBatchSize-1,dataset:size()) do
        -- load new sample
        local idx = math.random(dataset:size())
        local sample = dataset[idx]
        inputs[k] = sample[1]:clone()
        cond_inputs1[k] = sample[2]:clone()
        cond_inputs2[k] = sample[3]:clone()
        k = k + 1
      end
      targets[{{1,dataBatchSize}}]:fill(1)
      -- (1.2) Sampled data
      noise_inputs:uniform(-1, 1)
      for i = dataBatchSize+1,opt.batchSize do
        local idx = math.random(dataset:size())
        local sample = dataset[idx]
        cond_inputs1[i] = sample[2]:clone()
        cond_inputs2[i] = sample[3]:clone()
      end
      local samples = model_G:forward({noise_inputs[{{dataBatchSize+1,opt.batchSize}}], cond_inputs1[{{dataBatchSize+1,opt.batchSize}}], cond_inputs2[{{dataBatchSize+1,opt.batchSize}}]})
      for i = 1, dataBatchSize do
        inputs[k] = samples[i]:clone()
        k = k + 1
      end
      targets[{{dataBatchSize+1,opt.batchSize}}]:fill(0)

      optim.sgd(fevalD, parameters_D, sgdState_D)
    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:uniform(-1, 1)
    for i = 1,opt.batchSize do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      cond_inputs1[i] = sample[2]:clone()
      cond_inputs2[i] = sample[3]:clone()
    end
    targets:fill(1)
    optim.sgd(fevalG, parameters_G, sgdState_G)

    -- disp progress
    xlua.progress(t, N)
  end -- end for loop over dataset

  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()

  -- save/log current net
  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, 'conditional_adversarial.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D = model_D, G = model_G, E = model_E, opt = opt})
  end

  -- next epoch
  epoch = epoch + 1
end

-- test function
function adversarial.test(dataset, N)
  local time = sys.clock()
  local N = N or dataset:size()

  local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
  local targets = torch.Tensor(opt.batchSize)
  local noise_inputs 
  if type(opt.noiseDim) == 'number' then
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim)
  else
    noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  end
  local cond_inputs1
  local cond_inputs2
  if type(opt.condDim1) == 'number' then
    cond_inputs1 = torch.Tensor(opt.batchSize, opt.condDim1)
  else
    cond_inputs1 = torch.Tensor(opt.batchSize, opt.condDim1[1], opt.condDim1[2], opt.condDim1[3])
  end
  if type(opt.condDim2) == 'number' then
    cond_inputs2 = torch.Tensor(opt.batchSize, opt.condDim2)
  else
    cond_inputs2 = torch.Tensor(opt.batchSize, opt.condDim2[1], opt.condDim2[2], opt.condDim2[3])
  end


  print('\n<trainer> on testing set:')
  for t = 1,N,opt.batchSize do
    -- display progress
    xlua.progress(t, N)

    ----------------------------------------------------------------------
    -- (1) Real data
    local targets = torch.ones(opt.batchSize)
    local k = 1
    for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      local idx = math.random(dataset:size())
      local sample = dataset[idx]
      inputs[k] = sample[1]:clone()
      cond_inputs1[k] = sample[2]:clone()
      cond_inputs2[k] = sample[3]:clone()
      k = k + 1
    end
    local preds = model_D:forward({inputs, cond_inputs1, cond_inputs2}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end

    ----------------------------------------------------------------------
    -- (2) Generated data (don't need this really, since no 'validation' generations)
    noise_inputs:uniform(-1, 1)
    local c = 1
    for i = 1,opt.batchSize do
      sample = dataset[math.random(dataset:size())]
      cond_inputs1[i] = sample[2]:clone()
      cond_inputs2[i] = sample[3]:clone()
    end
    local samples = model_G:forward({noise_inputs, cond_inputs1, cond_inputs2})
    local targets = torch.zeros(opt.batchSize)
    local preds = model_D:forward({samples, cond_inputs1, cond_inputs2}) -- get predictions from D
    -- add to confusion matrix
    for i = 1,opt.batchSize do
      local c
      if preds[i][1] > 0.5 then c = 2 else c = 1 end
      confusion:add(c, targets[i] + 1)
    end
  end -- end loop over dataset

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()

  local samples = model_G.output
  local to_plot = {}
  for i=1,opt.batchSize do
    to_plot[i] = samples[i]:float()
  end
  local fname = paths.concat(opt.save, 'epoch-' .. epoch .. '.png')
  torch.setdefaulttensortype('torch.FloatTensor')
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true})
  if opt.gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end
  return cond_inputs
end

return adversarial


]]--