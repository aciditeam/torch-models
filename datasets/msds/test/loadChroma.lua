-- Test script: open an h5 file and load its beat-aligned chromagram

local mgs_getters = require '../beatAlignedFeats'

h5read = mgs_getters.open_h5_file_read('./test_data/TRAAAAW128F429D538.h5')
bt_chromas = mgs_getters.get_btchromas(h5read)

return bt_chromas
