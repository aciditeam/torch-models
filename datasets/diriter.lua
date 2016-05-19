-- Directory tree-structure iterator
--
-- 
-- original code by AlexanderMarinov
-- Compatible with Lua 5.1 (not 5.0).
-- via <http://lua-users.org/wiki/DirTreeIterator>
--
-- 
-- Some modifications to make it work

require 'lfs'

local diriter = {}

function diriter.dirtree(dir)
   assert(dir and dir ~= "", "directory parameter is missing or empty")
   if string.sub(dir, -1) == "/" then
      dir=string.sub(dir, 1, -2)
   end

   _, obj_iter = lfs.dir(dir)
   local diriters = {obj_iter}
   local dirs = {dir}
   
   return function()
      repeat
	 local entry = diriters[#diriters]:next()
	 if entry then
	    if entry ~= "." and entry ~= ".." then
	       local filename = table.concat(dirs, "/").."/"..entry
	       local attr = lfs.attributes(filename)
	       if attr.mode == "directory" then
		  table.insert(dirs, entry)
		  _, obj_iter_subfolder = lfs.dir(filename)
		  table.insert(diriters, obj_iter_subfolder)
	       end
	       return filename, attr
	    end
	 else
	    table.remove(dirs)
	    table.remove(diriters)
	 end
      until #diriters==0  -- Captures empty obj_iterator in lfs.dir(dir) after
      -- end of iteration
   end
end

local function is_suffix(str, suffix)
   return suffix=='' or string.sub(str,-string.len(suffix))==suffix
end

-- Convert iterator as returned by diriter.dirtree to array
-- Specialized for diriter.dirtree iterator, returning only
function diriter.to_array(filter_suffix, ...)
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

return diriter
