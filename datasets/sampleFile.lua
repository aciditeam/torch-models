-- Sample a random filename from a directory structure

local diriter = require './diriter'

local M = {}

-- Extract a subset from elems indexed by table indexes
local function get_subset(elems, indexes)
   local subset = {}
   for i=1, #indexes do
      table.insert(subset, elems[indexes[i]]) 
   end
   return subset
end

-- Extract a subset from elems indexed by table indexes
function M.get_random_subset(elems, sample_size)
   local indexes = torch.randperm(#elems):sub(1, sample_size):totable()
   local sample = get_subset(elems, indexes)
   return sample
end

-- Initialize filename sampling function for the chosen folder.
--
-- The returned function outputs a new set of unique, random filenames of
-- given size on each call.
-- 
-- Input:
--  * root_path, a string: the name of the root of the folder structure
--  * filter_suffix, a string, optional: keep only filenames with this suffix
--   (defaults to '', no filtering)
--
function M.get_generator(root_path, filter_suffix)
   local suffix = suffix or ''
   local dir_iterator = diriter.dirtree(root_path)
   local filenames = diriter.to_array(filter_suffix, dir_iterator)
   local files_n = #filenames

   return function(sample_size)
      local sample_size = sample_size or 1

      -- draw sample_size unique filenames
      -- TODO: a full files_n permutation is sub-optimal, but how
      -- to make it better?
      local sample = M.get_random_subset(filenames, sample_size)
      return sample
   end
end

return M
