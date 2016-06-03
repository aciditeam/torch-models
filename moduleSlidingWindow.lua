----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Auxiliary modules
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'rnn'
require 'torch'
require 'modelClass'
local nninit = require 'nninit'

----------------------------------------------------------------------
-- Sliding window module
--
-- This module takes a 1-dimensional time series as input and outputs all its sub-sequences of given size every given step
-- The optional tensOut (default false) decides if the output is either one large Tensor or a table of Tensors
-- The number of subsequences can be limited with parameter nf
----------------------------------------------------------------------

local SlidingWindow, parent = torch.class('nn.SlidingWindow','nn.Module')

function SlidingWindow:__init(tDim, size, step, nf, tensOut)
   parent.__init(self)
   self.tDim = tDim or 1
   self.size = size or 16
   self.step = step or 1
   self.nfeatures = nf or 1e9
   self.tensOut = tensOut or false
end

function SlidingWindow:updateOutput(input)
   local rep = torch.ceil((input:size(self.tDim) - self.size + 1) / self.step)
   local sz = torch.LongStorage(input:dim()+1)
   local currentOutput= {}
   if self.tensOut then currentOutput = torch.Tensor(rep, self.size, input:size(2)) end
   for i=1,rep do
      local slice = input:narrow(self.tDim, ((i - 1) * self.step + 1), self.size)
      currentOutput[i] = slice
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
