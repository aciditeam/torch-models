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
require "optim"
local nninit = require 'nninit'

local GaussianReparam, parent = torch.class('nn.GaussianReparam', 'nn.Module')
--Based on JoinTable
function GaussianReparam:__init(dimension,noiseparam)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
  self.noiseparam = noiseparam or 0.05
  self.train = true
  self.KL = 0
end 

--Forward pass
function GaussianReparam:updateOutput(input)  
  if input[1]:dim()==1 then  --SGD setting 
    if not self.dimension then self.dimension = input[1]:size(1) end
  elseif input[1]:dim()==2 then --Batch setting 
    if not self.dimension then self.dimension = input[1]:size(2) end 
  else
    error('Input must be a vector or a matrix')
  end 
  --Treat input[2] as log sigma^2
  self.eps = torch.randn(input[1]:size()):typeAs(input[1])
  local noise = torch.randn(input[1]:size()):typeAs(input[1])

  self.output = torch.exp(input[2]*0.5):cmul(self.eps):add(input[1])
  local kl = (input[2] + 1):mul(-1):add(torch.pow(input[1],2)):add(torch.exp(input[2]))
  self.KL = kl:sum()*0.5

  --Add noise to output during training 
  if not self.train then
    noise:fill(0)
  end
  self.output:add(noise*self.noiseparam)
    return self.output
end

--Backward pass
function GaussianReparam:updateGradInput(input, gradOutput)
  --Gradient with respect to mean
  self.gradInput[1]= gradOutput+input[1]
  --Gradient with respect to R
  self.gradInput[2]=torch.mul(input[2],0.5):exp():mul(0.5):cmul(self.eps):cmul(gradOutput)
  local grad_R = (torch.exp(input[2])-1)*0.5
  self.gradInput[2]:add(grad_R)
    return self.gradInput
end

local modelVariational, parent = torch.class('modelVariational', 'modelClass')

----------------------------------------------------------------------
-- Deep Generative Model trained using Stochastic Backpropagation
-- References :
-- Auto-Encoding Variational Bayes
-- http://arxiv.org/abs/1312.6114
-- Stochastic Backpropagation and Approximate Inference in Deep Generative Models
-- http://arxiv.org/abs/1401.4082
----------------------------------------------------------------------
function modelVariational:defineModel(structure, options)
  ---------------- Model Params. -----------
  local dim_stochastic = params.dim_stochastic or 100
  local nonlinearity   = params.nonlinearity or nn.ReLU
  --------- Recognition. Network -----------
  local var_inp = nn.Identity()()
  local dropped_inp = nn.Dropout(0.25)(var_inp)
  local q_1 = nonlinearity()(nn.Linear(dim_input,dim_hidden)(dropped_inp))
  local q_hid_1 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(q_1))
  local q_hid_2 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(q_hid_1))
  local mu  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
  local logsigma  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
  local reparam   = nn.GaussianReparam(dim_stochastic)
  -- print (reparam.KL)
  local z  = reparam({mu,logsigma})
  local var_model = nn.gModule({var_inp},{z})
  --------- Generative Network -------------
  local gen_inp = nn.Identity()()
  local hid1 = nonlinearity()(nn.Linear(dim_stochastic,dim_hidden)(gen_inp))
  local hid2 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(hid1))
  local hid3 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(hid2))
  local hid4 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(hid3))
  local reconstr = nn.Sigmoid()(nn.Linear(dim_hidden,dim_input)(hid4))
  local gen_model = nn.gModule({gen_inp},{reconstr})
  ----- Combining Models into Single MLP----
  local inp = nn.Identity()()
  mlp = nn.gModule({inp},{gen_model(var_model(inp))})
  --criterion = nn.BCECriterion()
  --criterion.sizeAverage = false
  return mlp; --criterion;
end

function modelVariational:definePretraining(structure, l, options)
  -- TODO
  return model;
end

function modelVariational:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder
  encoder:remove();
  return model.encoder;
end

function modelVariational:weightsInitialize(model)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', self.initialize);
    module:init('bias', self.initialize);
  end
  return model;
end

function modelVariational:weightsTransfer(model, trainedLayers)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#trainedLayers do
    -- Find equivalent in pre-trained layer
    preTrained = trainedLayers[l].encoder:findModules('nn.Linear');
    linearNodes[l].weight = preTrained[1].weight;
    linearNodes[l].bias = preTrained[1].bias;
  end
  -- Initialize the batch normalization layers
  for k,v in pairs(model:findModules('nn.BatchNormalization')) do
    v.weight:fill(1)
    v.bias:zero()
  end
  return model;
end

function modelVariational:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.pretrainType = 'ae';
  self.pretrain = true;
  self.dropout = 0.5;
end

function modelVariational:parametersRandom()
  -- All possible non-linearities
  self.distributions = {};
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
  self.distributions.batchNormalize = {true, false};
  self.distributions.pretrainType = {'ae', 'psd'};
  self.distributions.pretrain = {true, false};
  self.distributions.dropout = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
end

--[[

---------  Sample from Gen. Model---------
function getsamples()
  local p = gen_model:forward(torch.randn(batchSize,dim_stochastic):typeAs(data.train_x))
  local s = torch.gt(p:double(),0.5)
  local samples = {}
  local mean_prob = {}
  for i=1,batchSize do 
    samples[i] = s[i]:float():reshape(28,28)
    mean_prob[i] = p[i]:float():reshape(28,28)
  end
  return samples,mean_prob
end

---------  Evaluate Likelihood   ---------
function eval(dataset)
  mlp:evaluate()
  local probs = mlp:forward(dataset)
  local nll   = crit:forward(probs,dataset)
  mlp:training()
  return (nll+reparam.KL)/dataset:size(1),probs
end
--------- Stitch Images Together ---------
function stitch(probs,batch)
  local imgs = {}
  for i = 1,batchSize do 
    imgs[i] = torch.cat(probs[i]:float():reshape(28,28),batch[i]:float():reshape(28,28),2)
  end
  return imgs
end

-------------- Training Loop -------------
for epoch =1,5000 do 
    local upperbound = 0
  local trainnll = 0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train_x:size(1))
  if epoch==100 then config.learningRate = 5e-5 end
  if epoch > 30 then config.learningRate = math.max(config.learningRate / 1.000005, 0.000001) end
    --Make sure batches are always batchSize
    local N = data.train_x:size(1) - (data.train_x:size(1) % batchSize)
    local N_test = data.test_x:size(1) - (data.test_x:size(1) % batchSize)
  local probs 
    local batch = torch.Tensor(batchSize,data.train_x:size(2)):typeAs(data.train_x)
  -- Pass through data
    for i = 1, N, batchSize do
        xlua.progress(i+batchSize-1, data.train_x:size(1))

        local k = 1
        for j = i,i+batchSize-1 do
            batch[k] = data.train_x[ shuffle[j] ]:clone() 
            k = k + 1
        end

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            mlp:zeroGradParameters()
            probs = mlp:forward(batch)
            local nll = crit:forward(probs, batch)
            local df_dw = crit:backward(probs, batch)
            mlp:backward(batch,df_dw)
            local upperbound = nll  + reparam.KL 
      trainnll = nll + trainnll
            return upperbound, gradients+(parameters*0.05)
        end

        parameters, batchupperbound = optim.rmsprop(opfunc, parameters, config, state)
        upperbound = upperbound + batchupperbound[1]
    end
  
  --Save results
    if upperboundlist then
        upperboundlist = torch.cat(upperboundlist,torch.Tensor(1,1):fill(upperbound/N),1)
    else
        upperboundlist = torch.Tensor(1,1):fill(upperbound/N)
    end

    if epoch % 10  == 0 then
      print("\nEpoch: " .. epoch .. " Upperbound: " .. upperbound/N .. " Time: " .. sys.clock() - time)
    --Display reconstructions and samples
    img_format.title="Train Reconstructions"
    img_format.win = id_reconstr
    id_reconstr = disp.images(stitch(probs,batch),img_format)
    local testnll,probs = eval(data.test_x)
    local b_test = torch.zeros(100,data.test_x:size(2)) 
    local p_test = torch.zeros(100,data.test_x:size(2)) 
    local shufidx = torch.randperm(data.test_x:size(1))
    for i=1,100 do
      p_test[i] = probs[ shufidx[i] ]:double()
      b_test[i] = data.test_x[ shufidx[i] ]:double()
    end
    img_format.title="Test Reconstructions"
    img_format.win = id_testreconstr
    id_testreconstr = disp.images(stitch(p_test,b_test),img_format)
    img_format.title="Model Samples"
    img_format.win = id_samples
    local s,mp = getsamples()
    id_samples =  disp.images(s,img_format)
    img_format.title="Mean Probabilities"
    img_format.win = id_mp
    id_mp =  disp.images(mp,img_format)
    print ("Train NLL:",trainnll/N,"Test NLL: ",testnll)
        torch.save('save/parameters.t7', parameters)
        torch.save('save/state.t7', state)
        torch.save('save/upperbound.t7', torch.Tensor(upperboundlist))
    local s,mp = getsamples()
    torch.save('save/samples.t7',s)
    torch.save('save/mean_probs.t7',mp)
    end
end
--]]