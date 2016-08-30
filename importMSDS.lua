-- Million-Song Dataset import utilities 

local locals = require './local'

local M = {}

M.load = {}

M.load = require './datasets/msds/beatAlignedFeats'

-- List disfunctional files and ignore them
local blacklist = {}

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
   local h5Filename
   if type(h5) ~= 'string' then
      -- Assumption that h5 is an open hdf5 file
      h5Filename = hdf5.HDF5File.filename(h5)
   else
      h5Filename = h5
   end

   if is_suffix(h5Filename, '.dat') then
      -- Uses precomputed beat-aligned chromagram
      return torch.load(h5Filename)
   end

   if use_blacklist and exists(h5Filename, blacklist, is_suffix) then
      print("WARNING: This hdf5 file has been blacklisted as incompatible" ..
	       ", skipping it")
      return torch.zeros(1, 12)
   end
   
   return local_load.get_btchromas(h5)
end

-- Full Million Song Dataset location
M.path = locals.paths.msds

local alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

local indexes = {}
indexes['TRAIN'] = {1, 22}
indexes['VALID'] = {23, 24}
indexes['TEST'] = {25, 26}

local function make_paths(indexes)
   local paths = {}
   for i=indexes[1], indexes[2] do
      local char = alphabet:sub(i,i)
      table.insert(paths,  char .. '/')
   end
   return paths
end

local setTypes = {'TRAIN', 'VALID', 'TEST'}

-- Splitting is:
--  * Training subset:   data/A/.../ through data/V/.../ (inclusive)
--  * Validation subset: data/W/.../ through data/X/.../ (inclusive)
--  * Training subset:   data/Y/.../ through data/Z/.../ (inclusive)
-- (Around 10% each for validation and testing)
M.sets = {}
for _, setType in pairs(setTypes) do
   M.sets[setType] = make_paths(indexes[setType])
end

local function shallow_copy(tableIn)
   local tableCopy = {}
   for k, v in pairs(tableIn) do
      tableCopy[k] = v
   end
   return tableCopy
end

local function compose_suffixes(prefix, suffixes)
   local composed = {}
   for _, suffix in ipairs(suffixes) do
      table.insert(composed, prefix .. suffix)
   end
   return composed
end

-- A much smaller validation subset of ~5K files for faster training
--  * Validation subset:  data/W/A/.../ through data/W/C/.../ (inclusive)
local smallValidSubfolders = make_paths({1, 3})  -- {'A/', ...,  'C/'}
M.smallValid = compose_suffixes('W/', smallValidSubfolders)

-- A much smaller training subset of ~10K files for faster training
--  * Training subset:  data/A/A.../ through data/A/G/.../ (inclusive)
local smallTrainSubfolders = make_paths({1, 7})  -- {'A/', ...,  'G/'}
M.smallTrain = compose_suffixes('A/', smallTrainSubfolders)

----------------------------------------------------------------------
-- Million Song Dataset 10K-songs-subset parameters
----------------------------------------------------------------------

M.subset = {}

M.subset.path = locals.paths.msdsSubset

M.subset.train = {'A/'}

-- This validation subset contains 1217 examples
M.subset.valid = compose_suffixes('B/', {'A/', 'B/', 'C/', 'D/'})

-- This testing subset contains 1381 examples
M.subset.test = compose_suffixes('B/', {'E/', 'F/', 'G/', 'H/', 'I/'})

M.subset.sets =
   {TRAIN = M.subset.train, VALID = M.subset.valid,
    TEST = M.subset.test}

return M
