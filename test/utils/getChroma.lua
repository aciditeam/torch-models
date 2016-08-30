-- Testing utilities: get a chromagram

local msds = require '../../importMSDS'
local msdsSample = require '../../datasets/msds/sample'

local sampler = msdsSample.get_chroma_sampler(msds.subset.path)

local M = {}

function M.getChromagram()
   return sampler()[1]
end

return M
