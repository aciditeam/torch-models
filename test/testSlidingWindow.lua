-- Test sliding sliding window module

local slidingWindow = require '../moduleSlidingWindow.lua'
local getChroma = require './utils/getChroma'
require 'rnn'

local slicer = nn.SlidingWindow(1, 64, 64, 1e9, true)

local useSequencer = false

local sequence = getChroma.getChromagram()

if useSequencer then
   slicer = nn.Sequencer(slicer)
   sequence = sequence:view(sequence:size(1), 1, sequence:size(2)) 
end

print(sequence:size())

local slices = slicer:forward(sequence)

print(slices)
print(slices:size())
