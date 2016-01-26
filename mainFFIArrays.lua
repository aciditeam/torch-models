----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Avoinding Lua's limitations through FFI arrays
--
----------------------------------------------------------------------
local ffi             = require("ffi") 
local size_double     = ffi.sizeof("double")
local size_char       = ffi.sizeof("char")
local size_float      = ffi.sizeof("float")
ffi.cdef"void* malloc (size_t size);"
ffi.cdef"void free (void* ptr);"
local chunk_size      = 16384
  
  ---------------------------------------------------------
-- define a structure which contains a size and array of 
-- doubles where we dynamically allocate the array using 
-- malloc() Do it this way just in case we want to write C
-- code that needs the size.
-- ---------------------------------------------------------
local DArr = ffi.metatype(
  --               size,               array
  "struct{uint32_t size; double* a;}",
  -- add some methods to our array
  { __index = {
    done = function(self) 
      if self.size == 0  then
        return false
      else
        ffi.C.free(self.a)
        self.a = nil
        self.size = 0
        return true
      end
    end,
    -- copy data element into our externally managed array from the
    -- supplied src array.   Start copying src[beg_ndx], stop copying
    -- at src[end_ndx],  copy into our array starting at self.a[dest_offset]
    copy_in = function(self,  src, beg_ndx, end_ndx, dest_offset)
      -- Can not use mem_cpy because the source is likely
      -- a native lua array.
      print ("self=", self,  " beg_ndx=", beg_ndx, 
         " end_ndx=", end_ndx, "dest_offset=", dest_offset)
      local mydata = self.a
      local dest_ndx  = dest_offset
      for src_ndx = beg_ndx, end_ndx do
         mydata[dest_ndx] = src[src_ndx]
         dest_ndx = dest_ndx + 1
      end
    end,
    -- copy data elements out of our externally managed array to another
    -- array.  Start copying at self.a[beg_ndx] ,  stop copying at self.a[end_ndx]
    -- place elements in dest starting at dest[dest_offset] and working up.
    copy_out = function(self, dest, beg_ndx, end_ndx, dest_offset)
      -- Can not can use mem_cpy because the dest is likely
      -- a native lua array.
      local mydata = self.a
      local dest_ndx  = dest_offset
      for ndx = beg_ndx, end_ndx do
        dest[dest_ndx] = mydata[ndx]
        dest_ndx = dest_ndx + 1
      end
    end,  
      -- return true if I still have a valid data pointer.
    -- return false if I have already ben destroyed.
    is_valid = function(self)
      print("is_valid() size=", self.size, " self.a=", 
        self.a, " self=", self)
      return self.size ~= 0 and  self.a ~= nil
    end,
    
    fill = function(self, anum, start_ndx, end_ndx)
      if end_ndx == nil then
        end_ndx = self.size
      end
      if start_ndx == nil then
        start_ndx = 0
      end 
      local mydata = self.a
      for ndx = 1, end_ndx do
        mydata[ndx] = anum
      end
    end,  -- func fill
    },

    __gc = function(self) 
         self:done()
    end
  }
)  -- end Darr()

--------------------------------------
 --- Constructor for double arrays
 -------------------------------------
function double_array(leng)
   -- allocate the actual dynamic buffer.
   local size_in_bytes = (leng + 1) * size_double
   local adj_bytes = (math.floor(size_in_bytes / chunk_size) + 1) * chunk_size
   local adj_len  = math.floor(adj_bytes / size_double)
   local ptr = ffi.C.malloc(adj_bytes)
   if ptr == nil then
     return nil
   end
   return DArr( adj_len,  ptr)
end

--------------------------------------
 --- Constructor for float arrays
 -------------------------------------
function float_array(leng)
   -- allocate the actual dynamic buffer.
   local size_in_bytes = (leng + 1) * size_float
   local adj_bytes = (math.floor(size_in_bytes / chunk_size) + 1) * chunk_size
   local adj_len  = math.floor(adj_bytes / size_float)
   local ptr = ffi.C.malloc(adj_bytes)
   if ptr == nil then
     return nil
   end
   return DArr(adj_len, ptr)
end