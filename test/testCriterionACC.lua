----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Binary accuracy criterion testing function
--
----------------------------------------------------------------------

require 'unsup'
require 'optim'
require 'torch'
require 'nninit'
require 'nn'

require '../criterionAcc.lua'

torch.setdefaulttensortype('torch.FloatTensor')

-- Initialize data

local inputs = torch.rand(4, 12)  -- Random batch of 4 chromas
print('Input batch of chromagrams:')
print(inputs)

local offset = 0.1
local targets = inputs + offset  -- Slightly shifted batch
print('Target batch of chromagrams (shifted by ' .. -offset .. '):')
print(targets)

local threshold = 0.2
print('Threshold is : ' .. threshold)

local inputs_threshold = inputs:gt(threshold)
local targets_threshold = targets:gt(threshold)

print('Criterion will threshold and compare both batches')
print('Thresholded inputs:')
print(inputs_threshold)

print('Thresholded targets:')
print(targets_threshold)

print('Differences:')
print(inputs_threshold:ne(targets_threshold))

local manual_error = inputs_threshold:ne(targets_threshold):float():
   div(inputs_threshold:numel()):
   sum()
print('Manual error computation: ' .. manual_error)

local criterion = nn.binaryAccCriterion(threshold)

print('Criterion forward:')
local output = criterion:forward(inputs, targets)

print('Criterion error: ' .. output)

print('Criterion backward:')
local gradInput = criterion:backward(inputs, targets)
print('Gradient:')
print(gradInput)
