-- Test script: sample a random chromagram from the MSDataset and return it

local sampler = require '../million_song_db/sample.lua'
local parameters = require '../local_parameters'

local path_to_data = parameters.msds_path

return sampler.get_chroma_sampler(path_to_data)(10)
