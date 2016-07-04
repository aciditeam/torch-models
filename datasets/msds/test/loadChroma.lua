-- Test script: open an h5 file and load its beat-aligned chromagram

-- local mgs_getters = require '../beatAlignedFeats'

local filename = './test_data/TRAAAAW128F429D538.h5'

local h5read
local i = 0
local dataPath = 'analysis/beats_confidence';
while true do
   local hdf5 = require 'hdf5'

   -- if i%1000 == 0 then print(i) end
 
   h5read = hdf5.open(filename, 'r')
   h5read:read(dataPath):all()
   h5read:close()
   h5read = nil
   hdf5 = nil
   collectgarbage(); collectgarbage()

   print(collectgarbage('count'))
   -- i = i+1
end
   -- bt_chromas = mgs_getters.get_btchromas(h5read)

return bt_chromas
