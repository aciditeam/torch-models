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

modelDRLIM = {};

----------------------------------------------------------------------
-- Handmade DRLIM
----------------------------------------------------------------------
function modelDRLIM.defineDRLIM(in_size, network_dim, params)
  
  --Encoder 
  local encoder = nn.Sequential()
  encoder:add(nn.Linear(im_size,network_dim[1]))
  encoder:get(encoder:size()).bias:add(-encoder:get(encoder:size()).bias:min())
  encoder:add(nn.Threshold())
  encoder:add(nn.Linear(network_dim[1],network_dim[2])) 
  encoder:get(encoder:size()).bias:add(-encoder:get(encoder:size()).bias:min())
  encoder:add(nn.Threshold())
  encoder:add(nn.Linear(network_dim[2],network_dim[3])) 
  encoder:cuda()

  -- Full network --> Split two input images
  local model = nn.Sequential()

  -- Create the parallel siamese network and add it to the full network
  local encoder_siamese = nn.ParallelTable()
  encoder_siamese:add(encoder)
  encoder_siamese:add(encoder:clone('weight','bias','gradWeight','gradBias'))
  model:add(encoder_siamese) 

  -- Create the L2 distance function and add it to the full network
  local dist = nn.PairwiseDistance(2)
  model:add(dist)

  model:cuda()

  -- Criterion
  local criterion = nn.HingeEmbeddingCriterion(margin):cuda() 

  --Necessary (Torch bug) 
  dist.gradInput[1] = dist.gradInput[1]:cuda()
  dist.gradInput[2] = dist.gradInput[2]:cuda()

  return model, criterion, encoder;
end

--[[
train = function()
  local point_pairs = gen_epoch_data(kNN,M)
  local av_error = 0 
  local nsamples = 0

  for i = 1,point_pairs:size(1) do
    if (math.mod(i,100)==0 or i == point_pairs:size(1)) then 
      progress(i,point_pairs:size(1))
    end
    
    if (point_pairs[i][1]~=0 and point_pairs[i][2]~=0 and
         point_pairs[i][1]~=point_pairs[i][2]) then
      local data = {torch.FloatTensor(im_size), torch.FloatTensor(im_size)}
      data[1] = X[ point_pairs[i][1] ]:cuda()
      data[2] = X[ point_pairs[i][2] ]:cuda()
      local target = point_pairs[i][3]

      local feval = function(x)
        gradParameters:zero()
        local pred = model:forward(data)
        local err = criterion:forward(pred, target)
        av_error = av_error + err
        local grad = criterion:backward(pred, target)
        model:backward(data, grad)
        return err, gradParameters
      end

      -- optimize on current sample
      optimMethod(feval, parameters, optimState)

      nsamples = nsamples + 1
    end
  end
   
  av_error = av_error / nsamples 
  return av_error  
end
--]]