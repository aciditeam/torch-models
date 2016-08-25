-- Test sliding sliding window module

local slidingWindow = require '../moduleSlidingWindow.lua'
local getChroma = require './utils/getChroma'
require 'rnn'

local criterion = nn.MSECriterion()

local sequence = getChroma.getChromagram()

local slicer, testSequence
for multivariate in ipairs({false, true}) do
   if multivariate then
      print('Multivariate sequence test')
      slicer = nn.SlidingWindow(1, 16, 1, 12, true)
      testSequence = sequence
   else
      print('Monovariate sequence test')
      slicer = nn.SlidingWindow(1, 16, 1, 1, true)
      testSequence = sequence[{{}, 1}]
   end

   print('sequence:size() = ')
   print(sequence:size())

   local batch = sequence:view(sequence:size(1), 1, sequence:size(2))
   local slices = slicer:forward(batch)

   print('slices:size() = ')
   print(slices:size())

   local targets = slices
   slices = slices * 0.99  -- Slightly distort slices

   local err = criterion:forward(slices, targets)

   local gradOutputs = criterion:backward(slices, targets)
   local gradInputs = slicer:backward(sequence, gradOutputs)

   print('gradInputs = ')
   print(gradInputs)
end
