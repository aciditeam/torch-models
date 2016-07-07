-- Test script: open an h5 file and load its beat-aligned chromagram

local mgs_getters = require '../beatAlignedFeats'
local hdf5 = require 'hdf5'

local filename = './test_data/TRAAAAW128F429D538.h5'

local h5read = hdf5.open(filename, 'r')
local bt_chromas = mgs_getters.get_btchromas(h5read)

return bt_chromas
