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
-- REPEAT module
-- 
-- Simple module repeats the end of the sequence it is fed with
-- Useful to compute baselines
----------------------------------------------------------------------

local Repeat, parent = torch.class('nn.Repeat','nn.Module')

function Repeat:__init(outDuration, options)
   parent.__init(self)
   self.outDuration = outDuration
   self.useCuda = options.cuda or false
   self.tDim = options.tDim or 1
end

function Repeat:updateOutput(input)
   local inputDuration = input:size(self.tDim)
   local lastStep = input:select(self.tDim, inputDuration)

   local sizes = torch.LongStorage({1, 1, 1})
   sizes[self.tDim] = self.outDuration

   local currentOutput = lastStep:repeatTensor(sizes)

   if self.useCuda then
      currentOutput = currentOutput:cuda()
   end

   self.output = currentOutput
   return self.output
end

function Repeat:updateGradInput(input, gradOutput)
   error('This module is not meant for training')
end
