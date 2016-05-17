-- Sample a random filename from a directory structure

local diriter = require 'diriter'

local M = {}

local function is_suffix(str, suffix)
   return suffix=='' or string.sub(str,-string.len(suffix))==suffix
end

-- Convert iterator as returned by diriter.dirtree to array
-- Specialized for diriter.dirtree iterator, returning only
local function diriter_to_array(filter_suffix, ...)
   local arr = {}
   local i = 1
   for filename, attr in ... do
      if attr.mode == 'file' and is_suffix(filename, filter_suffix) then 
	 arr[i] = filename
	 i = i+1
      end
   end
   return arr
end

-- Extract a subset from elems indexed by table indexes
local function take_subset(elems, indexes)
   local subset = {}
   for i=1, #indexes do
      table.insert(subset, elems[indexes[i]]) 
   end
   return subset
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
   local filenames = diriter_to_array(filter_suffix, dir_iterator)
   local files_n = #filenames

   return function(sample_size)
      local sample_size = sample_size or 1

      -- draw sample_size unique filenames
      -- TODO: a full files_n permutation is sub-optimal, but how
      -- to make it better?
      local indexes = torch.randperm(files_n):sub(1, sample_size):totable()
      return take_subset(filenames, indexes)
   end
end

return M
