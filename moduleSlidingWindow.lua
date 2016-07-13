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
----------------------------------------------------------------------

local SlidingWindow, parent = torch.class('nn.SlidingWindow','nn.Module')

function SlidingWindow:__init(tDim, size, step, nf)
   parent.__init(self)
   self.tDim = tDim or 1
   self.size = size or 16
   self.step = step or 1
   self.nf = nf or 1e9
end

function SlidingWindow:updateOutput(input)
   print(input:size())
   local rep = torch.ceil((input:size(self.tDim) - self.size + 1) / self.step)
   local sz = torch.LongStorage(input:dim()+1)
   local currentOutput= {}
   
   local dims = input:size()
   if dims:size() > 1 then
      -- Multi-dimensional input
      currentOutput = torch.Tensor(self.size, rep, dims[2])
   else
      currentOutput = torch.Tensor(self.size, rep, 1)
   end
   
   for i=1,rep do
      currentOutput[{{}, {i}, {}}] = input:narrow(self.tDim, ((i - 1) * self.step + 1), self.size)
   end

   self.output = currentOutput
   return self.output
end

function SlidingWindow:updateGradInput(input, gradOutput)
   local slices = input:size(self.tDim)
   self.gradInput:resizeAs(input):zero()
   for i=1,#gradOutput do 
      local currentGradInput = gradOutput[i];
      local curIdx = ((i - 1) * self.step + 1);
      if (self.tDim == 1) then        
        self.gradInput[{{curIdx, curIdx + self.size - 1}}]:add(currentGradInput);
      else 
        self.gradInput[{{}, {curIdx, curIdx + self.size - 1}}]:add(currentGradInput);
      end
   end
   return self.gradInput
end
