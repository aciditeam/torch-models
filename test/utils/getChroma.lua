-- Testing utilities: get a chromagram

local slidingWindow = require '../moduleSlidingWindow'
local msds = require '../importMSDS'
local msdsSample = require '../datasets/msds/sample'

local sampler = msdsSample.get_chroma_sampler(msds.subset.path)

function getChromagram()
   return sampler()[1]
end
