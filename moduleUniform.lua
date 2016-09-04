----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'rnn'
require 'unsup'
require 'optim'
require 'torch'
require 'modelClass'
local nninit = require 'nninit'

----------------------------------------------------------------------
-- UNIFORM sampling module
-- 
-- Simple module that performs uniform sampling in [0, 1]
-- Useful to compute baselines
----------------------------------------------------------------------

local Uniform, parent = torch.class('nn.Uniform','nn.Module')

function Uniform:__init(outDuration, outFeats, options)
   parent.__init(self)
   self.outDuration = outDuration
   self.outFeats = outFeats
   self.useCuda = options.cuda or false
   self.tDim = options.tDim or 1
end

function Uniform:updateOutput(input)
   local batchSize = input:size(2)
   
   currentOutput = torch.rand(self.outDuration, batchSize,
			      self.outFeats)
   if self.useCuda then
      currentOutput = currentOutput:cuda()
   end
   self.output = currentOutput
   return self.output
end

function Uniform:updateGradInput(input, gradOutput)
   error('This module is not meant for training')
end
