-- Check contents of targets for time-series prediction

local mytest = torch.TestSuite()
local tester = torch.Tester()

local _ = require 'moses'

local import = require '../importTSDataset.lua'
local msds = require '../importMSDS.lua'
require '../SequencerSlidingWindow.lua'

local datasetPath = msds.subset.path
local datasetSets = msds.subset.sets

local filter_suffix = '.h5'

local filenamesSets = import.import_sets_filenames(
   datasetPath, datasetSets, filter_suffix, false)

-- Test over filesNum different files
local filesNum = 100
local filenames = _.first(filenamesSets['TRAIN'], filesNum)

local f_load = msds.load.get_btchromas

local sequence = f_load(filenames[1])

local options = {}
options.tDim = 1
options.batchDim = 2
options.featsDim = 3
options.featsNum = sequence:size(2)
options.sliceSize = 16
options.predict = true
options.predictionLength = 2
options.paddingValue = 0

function mytest.test()
   for __, filename in pairs(filenames) do
      local slices = import.load_slice_filenames_tensor(
	 {filename}, f_load, options)
      
      local slicedSequence = slices['data']
      local predictionTargets = slices['targets']
      
      local batchSize = slicedSequence:size(options.batchDim)
      for sliceInd=2,batchSize do
	 local currentSlice = slicedSequence:select(2, sliceInd-1)
	 local currentSliceContinuation = slicedSequence:select(2, sliceInd):
	    narrow(1, 1, options.predictionLength)
	 local currentSlicePredictionTarget = predictionTargets:select(
	    2, sliceInd-1)

	 tester:eq(currentSlicePredictionTarget, currentSliceContinuation,
		   "Prediction and continuation for current slice" ..
		      "should be equal")
      end
   end
end

tester:add(mytest)
tester:run()
