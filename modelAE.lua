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

modelAE = {};

----------------------------------------------------------------------
-- Simple auto-encoder
-- (eventually tied weights)
----------------------------------------------------------------------
function modelAE.defineAE(inputSize, outputSize, params)
  -- encoder
  encoder = nn.Sequential();
  encoder:add(nn.Linear(inputSize,outputSize));
  encoder:add(nn.PReLU());
  encoder:add(nn.Diag(outputSize));
  -- decoder
  decoder = nn.Sequential();
  decoder:add(nn.Linear(outputSize,inputSize));
  -- tied weights
  if params.tied and not params.hessian then
    -- impose weight sharing
    decoder:get(1).weight = encoder:get(1).weight:t();
    decoder:get(1).gradWeight = encoder:get(1).gradWeight:t();
  elseif params.tied then
    print('==> Warning: weight sharing only supported with no hessian computation');
  end
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, params.beta);
  return model;
end

----------------------------------------------------------------------
-- Convolutional auto-encoder
----------------------------------------------------------------------
function modelAE.defineCAE(inputSize, params)
  -- params:
  conntable = nn.tables.full(params.filtersin, params.filtersout)
  kw, kh = params.kernelsize, params.kernelsize
  iw, ih = inputSize, inputSize
  -- connection table:
  local decodertable = conntable:clone()
  decodertable[{ {},1 }] = conntable[{ {},2 }]
  decodertable[{ {},2 }] = conntable[{ {},1 }]
  local outputFeatures = conntable[{ {},2 }]:max()
  -- encoder:
  encoder = nn.Sequential()
  encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
  encoder:add(nn.Tanh())
  encoder:add(nn.Diag(outputFeatures))
  -- decoder:
  decoder = nn.Sequential()
  decoder:add(nn.SpatialFullConvolutionMap(decodertable, kw, kh, 1, 1))
  -- complete model
  model = unsup.AutoEncoder(encoder, decoder, params.beta)
  -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
  -- dataset:conv()
  -- verbose
  return model
end

----------------------------------------------------------------------
-- Linear Predictive Sparse Coding (PSD)
----------------------------------------------------------------------
function modelAE.definePSD(inputSize, outputSize, params)
  -- encoder
  encoder = nn.Sequential()
  encoder:add(nn.Linear(inputSize,outputSize))
  encoder:add(nn.PReLU())
  encoder:add(nn.Diag(outputSize))
  -- decoder is L1 solution
  decoder = unsup.LinearFistaL1(inputSize, outputSize, params.lambda)
  -- PSD autoencoder
  model = unsup.PSD(encoder, decoder, params.beta)
  return model;
end

----------------------------------------------------------------------
-- Convolutional Predictive Sparse Coding (PSD)
----------------------------------------------------------------------
function modelAE.defineCPSD(inputSize, outputSize, params)
  -- params:
  conntable = nn.tables.full(params.filtersin, params.filtersout)
  kw, kh = params.kernelsize, params.kernelsize
  iw, ih = inputSize, inputSize
  -- connection table:
  local decodertable = conntable:clone()
  decodertable[{ {},1 }] = conntable[{ {},2 }]
  decodertable[{ {},2 }] = conntable[{ {},1 }]
  local outputFeatures = conntable[{ {},2 }]:max()
  -- encoder:
  encoder = nn.Sequential()
  encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
  encoder:add(nn.Tanh())
  encoder:add(nn.Diag(outputFeatures))
  -- decoder is L1 solution:
  decoder = unsup.SpatialConvFistaL1(decodertable, kw, kh, iw, ih, params.lambda)
  -- PSD autoencoder
  model = unsup.PSD(encoder, decoder, params.beta)
  -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
  -- dataset:conv()
  return model;
end