-- Test script: sample a random chromagram from the MSDataset and return it

local sampler = require '../sample'
local msds = require '../../../importMSDS'

local path_to_data = msds.subset.path

return sampler.get_chroma_sampler(path_to_data)(1000)
