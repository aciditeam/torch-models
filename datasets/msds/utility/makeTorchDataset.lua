-- Convert h5 to torch data files to avoid usage of memory-leaking torch-hdf5

local msds = require '../../../importMSDS.lua'
local diriter = require '../../diriter.lua'

local dirData = msds.subset.path
local filter_suffix = '.h5'

local filenames_iterator = diriter.dirtree(dirData)
local filenames_table = diriter.to_array(filter_suffix, filenames_iterator)

for _, filename in ipairs(filenames_table) do
   local filename_save = filename:gsub('.h5', '-beat_aligned_chroma.dat')
   
   local bt_chroma = msds.load.get_btchromas(filename)
   torch.save(filename_save, bt_chroma)
   collectgarbage(); collectgarbage()
end
