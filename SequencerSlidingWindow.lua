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
-- Sequencer sliding window module
-- Follows rnn's conventions for batch learning of multivariate time-series.
-- 
-- This module takes a multivariate time series as input and outputs
-- all its sub-sequences of given size every given step.
-- The optional tensOut (default true) decides if the output is either
-- one large Tensor, with dimensions sequenceLength x batchSize x featSize
-- following rnn's conventions, or a table of Tensors.
-- The number of subsequences can be limited with parameter nf.
----------------------------------------------------------------------

local SequencerSlidingWindow, parent = torch.class('nn.SequencerSlidingWindow','nn.Module')

function SequencerSlidingWindow:__init(tDim, size, step, nf)
   parent.__init(self)
   self.tDim = tDim or 1
   self.size = size or 16
   self.step = step or 1
   self.nfeatures = nf or 1e9
end

function SequencerSlidingWindow:updateOutput(input)
   assert(torch.isTensor(input), 'Input to SlidingWindow module must be a torch tensor')
   local inputDims = input:dim()
   assert(inputDims == 3, 'Expects an input with dimensions '..
	     'sequenceLength x batchSize x featSize')

   local rep = torch.ceil((input:size(self.tDim) - self.size + 1) / self.step)

   -- Batch-mode, following rnn's conventions
   local batchSize = input:size(2)
   local featSize = input:size(3)
   -- local currentOutput = torch.zeros(self.size, rep * batchSize, featSize)

   -- Initial slice
   local currentOutput = input:narrow(self.tDim, 1, self.size)
   
   for i=2,rep do
      local slice = input:narrow(self.tDim, ((i - 1) * self.step + 1), self.size)
      currentOutput = currentOutput:cat(slice, 2)
   end

   self.output = currentOutput
   return self.output
end

function SequencerSlidingWindow:updateGradInput(input, gradOutput)
   error('Not correctly implemented yet!')

   local slicesNum = input:size(self.tDim)
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

local M = {}

-- Register parameters to a hyper-parameters structure for optimization  
function M.registerParameters(hyperParams, maxSize, maxStep)
   local maxSize = maxSize or 32
   local maxStep = maxStep or 32
   
   hyperParams:registerParameter('slidingWindow', 'bool');
   hyperParams:registerParameter('slidingWindowSize', 'int', {1, maxSize});
   hyperParams:registerParameter('slidingWindowStep', 'int', {1, maxStep});
end

function M.getParameters(hyperParams)
   local parameters = {}
   parameters.useSlidingWindow = hyperParams:getCurrentParameter(slidingWindow)
   parameters.slidingWindowSize = hyperParams:getCurrentParameter(slidingWindowSize)
   parameters.slidingWindowStep = hyperParams:getCurrentParameter(slidingWindowStep)
   return parameters
end

return M
