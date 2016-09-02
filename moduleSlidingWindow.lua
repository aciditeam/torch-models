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
-- Sliding window module
--
-- This module takes a featsNum-dimensional time series as input and outputs
--   all its sub-sequences of given size every given step
-- The number of subsequences can be limited with parameter nf
-- 
-- Output of the module is of the form:
--  * a table of (winSize x nFeats) Tensors if tensOut is false
--  * a Tensor of dimensions (nWins x winSize x nFeats) otherwise
----------------------------------------------------------------------

local SlidingWindow, parent = torch.class('nn.SlidingWindow','nn.Module')

function SlidingWindow:__init(tDim, size, step, nFeats,
			      tensOut, cuda, nf)
   parent.__init(self)
   self.tDim = tDim or 1
   self.size = size or 16
   self.step = step or 1
   self.nFeats = nFeats or 1
   self.tensOut = tensOut or false
   self.cuda = cuda or false
   self.nf = nf or 1e9
end

function SlidingWindow:updateOutput(input)
   local nWins = torch.ceil((input:size(self.tDim) - self.size + 1) / self.step)
   local batchSize = input:size(2)
   local currentOutput = {}

   if self.tensOut then
      if self.cuda then
	 currentOutput = torch.CudaTensor(nWins, batchSize, self.size, self.nFeats)
      else
	 currentOutput = torch.Tensor(nWins, batchSize, self.size, self.nFeats)
      end
   end

   for i=1,nWins do
      currentOutput[i] = input:narrow(self.tDim, ((i - 1) * self.step + 1),
				      self.size):transpose(1, 2):contiguous()
      if self.cuda and not self.tensOut then
	 currentOutput[i] = currentOutput[i]:cuda()
      end
   end
   
   self.output = currentOutput
   return self.output
end

function SlidingWindow:updateGradInput(input, gradOutput)
   local slices = input:size(self.tDim)
   if self.cuda then
      self.gradInput:resizeAs(input:cuda()):zero()
   else
      self.gradInput:resizeAs(input):zero()
   end
   
   if self.tensOut then
      nWins = gradOutput:size(1)
   else
      nWins = #gradOutput
   end
   
   for i=1,nWins do
      local currentGradInput
      currentGradInput = gradOutput[i]:transpose(1, 2)

      local curIdx = ((i - 1) * self.step + 1)
      local curWin = {curIdx, curIdx + self.size - 1}
      if (self.tDim == 1) then
	 if self.nFeats > 1 then
	    -- Multi-dimensional input
	    self.gradInput[{curWin, {}}]:add(currentGradInput)
	 else
	    self.gradInput[{curWin}]:add(currentGradInput)
	 end
      else
	 self.gradInput[{{}, curWin}]:add(currentGradInput)
      end
   end
   return self.gradInput
end
