----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Chromagram clustering
--
-- Uses code from the unsup library
-- 
----------------------------------------------------------------------

-- Cluster chromagrams for communication with OMax/Improtek

require 'unsup'
require 'optim'
require 'torch'
require 'nninit'

require 'moduleSlidingWindow'
require 'SequencerSlidingWindow'

local import_dataset = require './importTSDataset'
local ts_init = require './TSInitialize'

cmd = torch.CmdLine()
cmd:option('--useCuda', false, 'whether to enable CUDA processing')

-- parse input params
cmd_params = cmd:parse(arg)

local options = ts_init.get_options(cmd_params.useCuda)

-- Debug and printing parameters
-- Print current validation every ... analyzed files
options.printValidationRate = 200

-- RNN-library's batch conventions
options.tDim = 1
options.batchDim = 2
options.featsDim = 3

ts_init.set_globals(); ts_init.set_cuda(options)

-- All sequences will be sliced into sub-sequences of this duration
options.sliceSize = 128

-- Not all the dataset is loaded into memory in a single pass
options.datasetWindowSize = 300
options.datasetMaxEpochs = 20
options.datasetWindowStepSize = options.datasetWindowSize

-- Training parameters
options.batchSize = 128;

---------------------------------------------
-- Modified k-means (online) (via unsup)
---------------------------------------------
--
-- The k-means algorithm.
--
--   > x: is supposed to be an MxN matrix, where M is the nb of samples and each sample is N-dim
--   > k: is the number of kernels
--   > niter: the number of iterations
--   > batchsize: the batch size [large is good, to parallelize matrix multiplications]
--   > callback: optional callback, at each iteration end
--   > verbose: prints a progress bar...
--
--   < returns the k means (centroids) + the counts per centroid
--
function onlineKmeans(filenames, k, f_load, niter, batchsize, callback, verbose)
   -- args
   local help = 'centroids,count = unsup.kmeans(Tensor(npoints,dim), k [, niter, batchsize, callback, verbose])'
   filenames = filenames or error('missing argument: ' .. help)
   k = k or error('missing argument: ' .. help)
   niter = niter or 1
   batchsize = batchsize or math.min(1000, #filenames)
   
   -- local examplesNum = 0
   -- -- Get number of chromagrams in dataset
   -- print('Computing total number of chromas in dataset')
   -- for file_i, file in ipairs(filenames) do
   --    if verbose then xlua.progress(file_i, #filenames) end
   --    local sequence = f_load(file)
   --    examplesNum = examplesNum + sequence:size(1)
   -- end

   -- resize data
   local featsNum = f_load(filenames[1]):size(2) or error(
      'Chosen file could not be loaded')
   local k_size = torch.Tensor({k, featsNum}):long():storage()
   -- if x:dim() > 2 then
   --    x = x:reshape(x:size(1), x:nElement()/x:size(1))
   -- end

   -- some shortcuts
   local sum = torch.sum
   local max = torch.max
   local pow = torch.pow

   -- dims
   local nsamples = examplesNum or print('WARNING: nsamples not initialized')
   local ndims = featsNum

   -- initialize means
   -- local x2 = sum(pow(x,2),2)  -- Deffered to online computation
   local centroids = torch.Tensor(k,ndims):normal()
   for i = 1,k do
      centroids[i]:div(centroids[i]:norm())
   end
   local totalcounts = torch.zeros(k)
   
   -- callback?
   if callback then callback(0,centroids:reshape(k_size),totalcounts) end

   local no_overlap = true
   
   -- do niter iterations
   for i = 1,niter do
      print('Iteration number ' .. i .. ' of ' .. niter)
      for slice, file_position in import_dataset.get_sliding_window_iterator(
	 filenames, f_load, options, no_overlap) do
	 -- progress
	 if verbose then xlua.progress(file_position, #filenames) end
	 
	 local sliceData = slice['data']
	 sliceData = sliceData:reshape(sliceData:size(options.tDim) *
					  sliceData:size(options.batchDim),
				       sliceData:size(options.featsDim))
	 
	 local sliceData2 = sum(pow(sliceData,2),2)
	 
	 -- sums of squares
	 local c2 = sum(pow(centroids,2),2)*0.5

	 -- init some variables
	 local summation = torch.zeros(k,ndims)
	 local counts = torch.zeros(k)
	 local loss = 0

	 -- process batch
	 for i = 1,sliceData:size(1),batchsize do
	    -- indices
	    local lasti = math.min(i+batchsize-1,sliceData:size(1))
	    local m = lasti - i + 1

	    -- k-means step, on minibatch
	    local batch = sliceData[{ {i,lasti},{} }]
	    local batch_t = batch:t()
	    local tmp = centroids * batch_t
	    for n = 1,(#batch)[1] do
	       tmp[{ {},n }]:add(-1,c2)
	    end
	    local val,labels = max(tmp,1)
	    loss = loss + sum(sliceData2[{ {i,lasti} }]*0.5 - val:t())

	    -- count examplars per template
	    local S = torch.zeros(m,k)
	    for i = 1,(#labels)[2] do
	       S[i][labels[1][i]] = 1
	    end
	    summation:add( S:t() * batch )
	    counts:add( sum(S,1) )
	 end

	 -- normalize
	 for i = 1,k do
	    if counts[i] ~= 0 then
	       centroids[i] = summation[i]:div(counts[i])
	    end
	 end

	 -- total counts
	 totalcounts:add(counts)

	 -- callback?
	 if callback then 
	    local ret = callback(i,centroids:reshape(k_size),totalcounts) 
	    if ret then break end
	 end
	 collectgarbage(); collectgarbage()
      end
   end
   -- done
   return centroids:reshape(k_size),totalcounts   
end

---------------------------------------------
-- Compute clustering
---------------------------------------------

local msds = require './importMSDS'

-- local _, filenamesSets = ts_init.import_data(baseDir, setList, options)
local filter_suffix = '.dat'  -- use '.dat' for precomputed chromas
local filenamesSets, datasetFolders = import_dataset.import_sets_filenames(msds.subset.path,
									   msds.subset.sets,
									   filter_suffix)

local filenames = filenamesSets['TRAIN']
local dataset = datasetFolders['TRAIN']

local function subrange(elems, start_idx, end_idx)
   local sub_elems = {}
   for i=start_idx, end_idx do
      table.insert(sub_elems, elems[i])
   end
   return sub_elems
end
   
local saveFolder = '/Users/bazin/work/machine_learning/clusterings/msds-chromagrams/'

local f_load = msds.load.get_btchromas
local niter = options.datasetMaxEpochs
local batchSize = options.batchSize
local callback = nil
local verbose = true

-- Write a table to a file (not recursive)
--
-- Input:
--  * fd, an open file descriptor
--  * tableIn, the table to write
--  * mainTableName, 
local function writeTable(fd, mainTableName, tableIn)
   fd:write(mainTableName .. ':\n')
   for key, val in pairs(tableIn) do
      fd:write('\t' .. key .. ': ' .. tostring(val) .. '\n')
   end
   fd:write('\n\n')
   fd:flush()
end

for _, k in ipairs({10, 20, 50, 100}) do
   print('\nSTART: Computing clustering with ' .. k .. ' clusters\n')

   -- Unique identifier to save clusterings for future use
   -- (format: year_month_day-hour_minute_second)
   local session_date = os.date('%Y%m%d-%H%M%S', os.time())
   local savePrefix = saveFolder .. session_date .. '-k_'.. k .. '-'
   local fd_options = assert(io.open(savePrefix .. 'options.txt', 'w'))

   writeTable(fd_options, 'Options', options)
   
   local kmeansOptions = {k = k, niter = niter, batchSize = batchSize,
			  files = dataset}
   writeTable(fd_options, 'Kmeans options', kmeansOptions)

   fd_options:close()
   
   centroids, totalCounts = onlineKmeans(filenames, k, f_load, niter, batchSize,
					 callback, verbose)
   
   local filename_centroids = savePrefix .. 'centroids.dat'
   torch.save(filename_centroids, centroids)
   
   local filename_totalCounts = savePrefix .. 'totalcounts.dat'
   torch.save(filename_totalCounts, totalCounts)

   collectgarbage(); collectgarbage()
end
