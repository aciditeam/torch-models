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
require 'torch'
require 'nninit'
require 'modelClass'
require 'modelCNN'

----------------------------------------------------------------------
--  Torch Linear Unit with Orthogonal Weight Initialization
--  Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
--  http://arxiv.org/abs/1312.6120
--  Copyright (C) 2015 John-Alexander M. Assael (www.johnassael.com)
----------------------------------------------------------------------
local LinearO, parent = torch.class('nn.LinearO', 'nn.Linear')

function LinearO:__init(inputSize, outputSize)
    parent.__init(self, inputSize, outputSize)
    self:reset()
end

function LinearO:reset()
    initScale = 1.1 -- math.sqrt(2)
    -- Random matrices
    local M1 = torch.randn(self.weight:size(1), self.weight:size(1))
    local M2 = torch.randn(self.weight:size(2), self.weight:size(2))
    -- Minimum dimensions
    local n_min = math.min(self.weight:size(1), self.weight:size(2))
    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)
    self.weight:copy(Q1:narrow(2,1,n_min) * Q2:narrow(1,1,n_min)):mul(initScale)
    self.bias:zero()
end


local modelDDCNN, parent = torch.class('modelDDCNN', 'modelCNN')

function modelDDCNN:defineModel(structure, options)
  opt.latent_dims = 2
  local enc_dims = 100
  local trans_dims = 100
  -- Model Specific parameters
  local f_maps_1 = 32
  local f_size_1 = 5
  local f_maps_2 = 32
  local f_size_2 = 5
  local f_maps_3 = 32
  local f_size_3 = 3
  -- Encoder
  encoder = nn.Sequential()
  encoder:add(nn.Reshape(1, structure.nInputs))
  encoder:add(nn.SpatialConvolutionMM(1, f_maps_1, f_size_1, f_size_1))
  encoder:add(nn.ReLU())
  encoder:add(nn.SpatialMaxPooling(2,2,2,2))
    -- Layer 2
    encoder:add(nn.SpatialConvolutionMM(f_maps_1, f_maps_2, f_size_2, f_size_2))
    encoder:add(nn.ReLU())
    encoder:add(nn.SpatialMaxPooling(2,2,2,2))
    -- Layer 3
    encoder:add(nn.SpatialConvolutionMM(f_maps_2, f_maps_3, f_size_3, f_size_3))
    encoder:add(nn.ReLU())
    -- Final layers 
    encoder:add(nn.Reshape(f_maps_3*5*5))
    encoder:add(nn.LinearO(f_maps_3*5*5, enc_dims))
    encoder:add(nn.ReLU())
    encoder:add(nn.LinearO(enc_dims, enc_dims))
    encoder:add(nn.ReLU())
    encoder:add(nn.LinearO(enc_dims, opt.latent_dims))
    -- Decoder
    decoder = nn.Sequential()
    decoder:add(nn.LinearO(opt.latent_dims, enc_dims))
    decoder:add(nn.ReLU())
    decoder:add(nn.LinearO(enc_dims, enc_dims))
    decoder:add(nn.ReLU())    
    decoder:add(nn.LinearO(enc_dims, f_maps_3*6*6))
    decoder:add(nn.ReLU())
    decoder:add(nn.Reshape(f_maps_3, 6, 6))
    -- Layer 3
    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolutionMM(f_maps_3, f_maps_3, f_size_3-1, f_size_3-1))
    decoder:add(nn.ReLU())
    -- Layer 2
    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolutionMM(f_maps_3, f_maps_2, f_size_2-1, f_size_2-1))
    decoder:add(nn.ReLU())
    -- Layer 1
    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolutionMM(f_maps_2, f_maps_1, f_size_2, f_size_2))
    decoder:add(nn.ReLU())
    -- Start layer
    decoder:add(nn.SpatialUpSamplingNearest(2))
    decoder:add(nn.SpatialConvolutionMM(f_maps_1, 1, f_size_2, f_size_2))
    decoder:add(nn.Sigmoid())
    -- Clone enc-dec
    local encoder2 = encoder:clone("weight", "bias", "gradWeight", "gradBias")
    local decoder2 = decoder:clone("weight", "bias", "gradWeight", "gradBias")
    -- Define model
    local x_t_prev = nn.Identity()():annotate{name = 'x_t_prev'}
    local x_t = nn.Identity()():annotate{name = 'x_t'}
    local u_t = nn.Identity()():annotate{name = 'u_t'}
    -- Define Encoder Module
    local z_t_prev = encoder2(x_t_prev):annotate{name = 'z_t_prev'}
    local z_t = encoder(x_t):annotate{name = 'z_t'}
    -- Transition layer
    trans = nn.Sequential()
    trans:add(nn.LinearO(opt.action_size+opt.latent_dims*2, trans_dims))
    trans:add(nn.ReLU())
    trans:add(nn.LinearO(trans_dims, trans_dims))
    trans:add(nn.ReLU())
    trans:add(nn.LinearO(trans_dims, opt.latent_dims))
    local dynamics_all = trans(nn.JoinTable(2)({z_t_prev, z_t, nn.Reshape(opt.action_size)(u_t)})):annotate{name = 'dynamics'}
    -- Define Output
    local decoder_x_t_next = decoder(dynamics_all):annotate{name = 'decoder_x_t_next'}
    local decoder_x_t_cur = decoder2(z_t):annotate{name = 'decoder_x_t_cur'}
    -- Create complete model
    model = nn.gModule({x_t_prev, x_t, u_t}, {z_t_prev, z_t, dynamics_all, decoder_x_t_cur, decoder_x_t_next})
    return model
end

function modelMLP:definePretraining(inS, outS, options)
  --[[ Encoder part ]]--
  encoder = nn.Sequential();
  -- Linear transform
  encoder:add(nn.Linear(inS,outS));
  -- Batch normalization
  if self.batchNormalize then encoder:add(nn.BatchNormalization(outS)); end
  -- Non-linearity
  encoder:add(self.nonLinearity());
  -- Dropout
  if self.dropout then encoder:add(nn.Dropout(self.dropout)) end
  encoder:add(nn.Diag(outS));
  -- decoder
  decoder = nn.Sequential();
  decoder:add(nn.Linear(outS,inS));
  -- impose weight sharing
  decoder:get(1).weight = encoder:get(1).weight:t();
  decoder:get(1).gradWeight = encoder:get(1).gradWeight:t();
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, options.beta);
  return model;
end

function modelMLP:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  return model.encoder;
end

function modelMLP:weightsInitialize(model)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,#linearNodes do
    module = linearNodes[l];
    module:init('weight', self.initialize);
    module:init('bias', self.initialize);
  end
  return model;
end

function modelMLP:weightsTransfer(model, trainedLayers)
  -- Find only the linear modules
  linearNodes = model:findModules('nn.Linear')
  for l = 1,trainedLayers do
    -- Find equivalent in pre-trained layer
    preTrained = trainedLayers[l]:findModules('nn.Linear');
    linearNodes[l].weight = preTrained[1].weight;
    linearNodes[l].bias = preTrained[1].bias;
  end
  return model;
end

function modelMLP:parametersDefault()
  self.initialize = nn.kaiming;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = false;
  self.pretrain = false;
  self.dropout = 0.5;
end

function modelMLP:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end

-- Function to perform unsupervised training on a sub-model
function modelClass:unsupervisedTrain(model, unsupData, options)
  return unsupervisedTrain(model, unsupData, options);
end

-- Function to perform supervised training on the full model
function modelClass:supervisedTrain(model, data, options)
  return supervisedTrain(model, data, options);
end

-- Function to perform supervised testing on the model
function modelClass:supervisedTest(model, data, options)
  return supervisedTest(model, data, options);
end

--[[

function train(dataset)

    g_create_batch(state_train)

    -- epoch tracker
    epoch = epoch or 0

    -- local vars
    local err = {all=0, bce=0, bce_1=0, mse=0}

    -- shuffle at each epoch
    local shuffle = torch.randperm(#dataset.batch):long()

    for t = 1,#dataset.batch do
        
        -- xlua.progress(t, #dataset.batch)

        -- create mini batch
        local batch_x_prev = dataset.batch[shuffle[t] ][1]
        local batch_x_cur = dataset.batch[shuffle[t] ][2]
        local batch_u = dataset.batch[shuffle[t] ][3]
        local batch_y = dataset.batch[shuffle[t] ][4]

        local batch_size = batch_y:size(1)

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            
            -- get new parameters
            if x ~= params then
                params:copy(x)
            end

            -- reset gradients
            gradParams:zero()
            
            -- reset errors
            local mse_err, bce_err, bce_1_err = 0, 0, 0
            
            local z_t_next_true = encoder:forward(batch_y)

            -- evaluate function for complete mini batch                                                
            local z_t_prev, z_t_cur, z_t_next, x_t, x_t_next = unpack(model:forward({batch_x_prev, batch_x_cur, batch_u}))  

            -- BCE x_t
            bce_err = bce_err + criterion:forward(x_t, batch_x_cur)
            local d_x_t = criterion:backward(x_t, batch_x_cur):clone()
            
            -- BCE x_t+1
            bce_1_err = bce_1_err + criterion:forward(x_t_next, batch_y)
            local d_x_t1 = criterion:backward(x_t_next, batch_y):clone()  
            
            -- MSE z_t+1
            mse_err = mse_err + criterion_mse:forward(z_t_next, z_t_next_true) * opt.lambda
            local d_z_t_next = criterion_mse:backward(z_t_next, z_t_next_true):clone():mul(opt.lambda)
            
            -- Backpropagate
            model:backward({batch_x_prev, batch_x_cur, batch_u}, {
                    torch.zeros(batch_size, opt.latent_dims),
                    torch.zeros(batch_size, opt.latent_dims),
                    torch.zeros(batch_size, opt.latent_dims),
                    d_x_t,
                    d_x_t1
                })
            
            local trans_in = torch.cat(torch.cat(z_t_prev, z_t_cur), batch_u)
            trans:forward(trans_in)
            trans:backward(trans_in, d_z_t_next)
            
            -- Accumulate errors
            err.mse = err.mse + mse_err
            err.bce = err.bce + bce_err
            err.bce_1 = err.bce_1 + bce_1_err
            err.all = err.all + bce_err + bce_1_err + mse_err
                        
            -- normalize gradients and f(X)
            local batcherr = (bce_err + bce_1_err + mse_err) / batch_size
            gradParams:div(batch_size)
                
            -- print(bce_err/batch_size, bce_1_err/batch_size, mse_err/batch_size)
                
            -- return f and df/dX
            return batcherr, gradParams
        end
        
        if batch_size > 0 then
            optim.adam(feval, params, optim_config)
            -- optim.adagrad(feval, params, optim_config)
            -- optim.rmsprop(feval, params, optim_config)
        end
        
    end
    
    -- Normalise errors
    err.all = err.all / (dataset.x:size(1) - 2)
    err.mse = err.mse / (dataset.x:size(1) - 2)
    err.bce = err.bce / (dataset.x:size(1) - 2)
    err.bce_1 = err.bce_1 / (dataset.x:size(1) - 2)
    
    epoch = epoch + 1

    return err
end

--]]