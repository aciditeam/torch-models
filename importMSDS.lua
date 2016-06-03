-- Million-Song Dataset import utilities 

local M = {}

M.load = {}

M.load = require './datasets/msds/beatAlignedFeats'

-- List disfunctional files and ignore them
local blacklist = {
   'TRAAYNQ128EF35922B.h5'
}

M.__blacklist = blacklist

local use_blacklist = false  -- Disable blacklist for debugging

local function exists(elem, elemsTable, predicate)
   for _, other_elem in ipairs(elemsTable) do
      if predicate(elem, other_elem) then return true end
   end
   return false
end

local function is_suffix(str, suffix)
   return suffix=='' or string.sub(str,-string.len(suffix))==suffix
end

local local_load = require './datasets/msds/beatAlignedFeats'
local hdf5 = require 'hdf5'

function M.load.get_btchromas(h5)
   if type(h5) ~= 'string' then h5 = hdf5.HDF5File.filename(h5) end
      
   if exists(h5, blacklist, is_suffix) and use_blacklist then
      print("WARNING: This hdf5 file has been blacklisted as incompatible" ..
	       ", skipping it")
      return torch.zeros(1, 12)
   end
   return local_load.get_btchromas(h5)
end

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
