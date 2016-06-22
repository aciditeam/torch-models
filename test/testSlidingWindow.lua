-- Test sliding sliding window module

local slidingWindow = require '../moduleSlidingWindow'
local getChroma = require './utilities/getChroma'

local slicer = nn.SlidingWindow(1, 64, 64, 1e9, true)

local sequence = getChroma.getChromagram()

print(sequence:size())

local slices = slicer:forward(sequence)

print(slices)
print(slices:size())
