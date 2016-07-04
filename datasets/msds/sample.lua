-- Sample an h5 file from the database and load its beat-aligned chromagram

local msds_getters = require './beatAlignedFeats'
local file_sample = require '../sampleFile'

local M = {}

-- Returns a chromagram sampled at random from the given dataset
-- The first dimension in this chromagram indexes time.
function M.get_chroma_sampler(root_path)
   local file_sampler = file_sample.get_generator(root_path, '.h5')

   return function(batch_size)
      local filenames = file_sampler(batch_size)

      local batch_size = #filenames
      local bt_chromas_batch = {}
      for i = 1, batch_size do
	 local bt_chromas = msds_getters.get_btchromas(filenames[i])

	 table.insert(bt_chromas_batch, bt_chromas)
      end
      return bt_chromas_batch
   end
end

return M
