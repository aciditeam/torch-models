-- Million-Song Dataset import utilities 

local M = {}

M.load = {}

M.load = require './datasets/msds/beatAlignedFeats'

-- Full Million Song Dataset location
-- M.path = './...'

----------------------------------------------------------------------
-- Million Song Dataset 10K-songs-subset parameters
----------------------------------------------------------------------

M.subset = {}

M.subset.path = '/data/Documents/machine_learning/datasets/mir/'..
   'MillionSongSubset/data/'

M.subset.training = {'A/'}

local function compose_suffixes(prefix, suffixes)
   local composed = {}
   for _, suffix in ipairs(suffixes) do
      table.insert(composed, prefix .. suffix)
   end
   return composed
end

-- This validation subset contains 1217 examples
M.subset.validation = compose_suffixes('B/', {'A/', 'B/', 'C/', 'D/'})

-- This testing subset contains 1381 examples
M.subset.testing = compose_suffixes('B/', {'E/', 'F/', 'G/', 'H/', 'I/'})

M.subset.sets =
   {TRAIN = M.subset.training, VALID = M.subset.validation,
    TEST = M.subset.testing}

return M
