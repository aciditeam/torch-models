----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Functions for data import
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'nn'
require 'torch'
require 'image'
require 'mainFFIArrays'

local preprocess = require './mainPreprocess'
local diriter = require './datasets/diriter'
local sampleFile = require './datasets/sampleFile'
local ucr = require './importUCR'
local msds = require './importMSDS'
local nninit = require 'nninit'

local moses = require 'moses'

local iterator = require './slidingIterator.lua'

local M = {}

----------------------------------------------------------------------
-- A real resampling function for time series
----------------------------------------------------------------------
function M.tensorResampling(data, destSize, type)
  -- Set the type of kernel
  local type = type or 'gaussian'
  -- Check properties of input data
  if data:dim() == 1 then
    data:resize(1, data:dim(1));
  end
  -- Original size of input
  inSize = data:size(2);
  -- Construct a temporal convolution object
  interpolator = nn.TemporalConvolution(inSize, destSize, 1, 1);
  -- Zero-out the whole weights
  interpolator.weight:zeros(destSize, inSize);
  -- Lay down a set of kernels
  for i = 1, destSize do
    if type == 'gaussian' then
      interpolator.weight[i] = image.gaussian1D(inSize, (1 / inSize), 1, true, i / destSize);
    else
      -- No handling of boundaries right now
      for j = math.max({i-kernSize, 1}),math.min({i+kernSize,destSize}) do
        -- Current position in kernel
        relIdx = (j - i) / kernSize;
        if type == 'bilinear' then
          interpolator.weight[i][j] = 1 - math.abs(relIdx);
        elseif type == 'hermite' then
	   interpolator.weight[i][j] =
	      (2 * (math.abs(x) ^ 3)) - (3 * (math.abs(x) ^ 2)) + 1;
        elseif type == 'lanczos' then
	   interpolator.weight[i][j] =
	      (2 * (math.abs(x) ^ 3)) - (3 * (math.abs(x) ^ 2)) + 1;
        end
      end
    end
  end
  -- print(interpolator.weight);
  return interpolator:forward(data);
end

----------------------------------------------------------------------
-- Helper functions
----------------------------------------------------------------------

local function make_structure(inputData, classes)
   -- Transform raw data to structure
   local structure = {
      data = inputData:float(),
      labels = torch.Tensor(classes),  -- classes may be nil
      mean = torch.Tensor(inputData:size(1)),
      std = torch.Tensor(inputData:size(1)),
      instances = function () return (data:size(1)) end,
      length = function () return (data:size(2)) end
   };
   -- Zero-mean unit-variance normalization
   for i = 1,(structure.data:size(1)) do
      structure.mean[i] = structure.data[i]:mean();
      structure.std[i] = structure.data[i]:std();
      structure.data[i] = (structure.data[i] - structure.mean[i]) / structure.std[i];
   end
   print('        = ' .. structure.data:size(1) .. ' instances of length ' .. structure.data:size(2));
   return structure
end

----------------------------------------------------------------------
-- Constructing a validation set for each dataset
----------------------------------------------------------------------
function M.construct_validation(sets, validPercent)
  -- For each dataset
  for v, k in pairs(sets) do
    -- Take only the train set
    local trainData = sets[v]["TRAIN"];
    print(' Separate ' .. trainData.data:size(1) .. ' instances.');
    -- Shuffle order of the set
    local shuffle = torch.randperm(trainData.data:size(1)):long();
    -- Number of instances to extract
    local nbInstances = trainData.data:size(1);
    local idValid = torch.round(trainData.data:size(1) * validPercent);
    -- Extract validation set
    local validData = {
      data = trainData.data.index(trainData.data, 1, shuffle[{{1,idValid}}]),
      labels = trainData.labels.index(trainData.labels, 1, shuffle[{{1,idValid}}]),
      mean = trainData.mean.index(trainData.mean, 1, shuffle[{{1,idValid}}]),
      std = trainData.std.index(trainData.std, 1, shuffle[{{1,idValid}}]),
      instances = function () return (data:size(1)) end,
      length = function () return (data:size(2)) end
    };
    -- Extract train set
    local curData = {
      data = trainData.data.index(trainData.data, 1, shuffle[{{idValid+1,nbInstances}}]),
      labels = trainData.labels.index(trainData.labels, 1, shuffle[{{idValid+1,nbInstances}}]),
      mean = trainData.mean.index(trainData.mean, 1, shuffle[{{idValid+1,nbInstances}}]),
      std = trainData.std.index(trainData.std, 1, shuffle[{{idValid+1,nbInstances}}]),
      instances = function () return (data:size(1)) end,
      length = function () return (data:size(2)) end
    };
    sets[v]["TRAIN"] = curData;
    sets[v]["VALID"] = validData;
    print('   - Train : ' .. curData.data:size(1));
    print('   - Valid : ' .. validData.data:size(1));
  end
  collectgarbage();
  return sets;
end

----------------------------------------------------------------------
-- Constructing a huge unsupervised dataset (only based on the "train" parts)
----------------------------------------------------------------------
function M.construct_unsupervised(sets, validPercent)
  -- Number of datasets
  local nbSets = #sets;
  local sizeSet = 0;
  local curSeries = 0;
  local seriesSize = 0;
  -- First pass for collecting sizes
  for v, k in pairs(sets) do
    -- Take only the train set
    local trainData = sets[v]["TRAIN"];
    sizeSet = sizeSet + trainData.data:size(1);
    seriesSize = trainData.data:size(2); 
  end
  -- We will only be interested in data
  finalData = {
    data = torch.Tensor(sizeSet, seriesSize)
    };
  -- Second pass to collect data
  for v, k in pairs(sets) do
    -- Take only the train set
    local trainData = sets[v]["TRAIN"];
    for t = 1,trainData.data:size(1) do
      finalData.data[curSeries+t] = trainData.data[t];
    end
    curSeries = curSeries + trainData.data:size(1);
  end
  local trainData = finalData;
  print(' Separate ' .. trainData.data:size(1) .. ' instances.');
  -- Shuffle order of the set
  local shuffle = torch.randperm(trainData.data:size(1)):long();
  -- Number of instances to extract
  local nbInstances = trainData.data:size(1);
  local idValid = torch.round(trainData.data:size(1) * validPercent);
  -- Extract validation set
  local validData = {
    data = trainData.data.index(trainData.data, 1, shuffle[{{1,idValid}}]),
    instances = function () return (data:size(1)) end,
    length = function () return (data:size(2)) end
  };
  -- Extract train set
  local curData = {
    data = trainData.data.index(trainData.data, 1, shuffle[{{idValid+1,nbInstances}}]),
    instances = function () return (data:size(1)) end,
    length = function () return (data:size(2)) end
  };
  finalData = {};
  finalData["TRAIN"] = curData;
  finalData["VALID"] = validData;
  print('   - Train : ' .. curData.data:size(1));
  print('   - Valid : ' .. validData.data:size(1));
  collectgarbage();
  return finalData;
end

----------------------------------------------------------------------
-- Perform data augmentation (manifold densification) for time series
--  * Noise, outlier and warping (for axioms of robustness)
--  * Eventually crop, scale (sub-sequence selections)
--  Should plot the corresponding manifold densification
----------------------------------------------------------------------
function M.data_augmentation(sets)
  -- First collect the overall maximum number of series
  local maxSize = 0;
  local totalSize = 0;
  local seriesSize = 0;
  -- First pass for collecting sizes
  for v, k in pairs(sets) do
    -- Take only the train set
    local trainData = sets[v]["TRAIN"];
    maxSize = (maxSize < trainData.data:size(1)) and trainData.data:size(1) or maxSize; 
    totalSize = totalSize + trainData.data:size(1);
    seriesSize = trainData.data:size(2); 
  end
  --maxSize = maxSize * 10;
  -- Iterate over the datasets
  for v, k in pairs(sets) do
    local trainData = sets[v]["TRAIN"];
    local curNbSeries = trainData.data:size(1);
    local seriesMissing = maxSize - curNbSeries;
    print("    * Densifying " .. v .. " with " .. seriesMissing .. " new series"); 
    if (seriesMissing > 0) then
      local newSeries = torch.zeros(seriesMissing, seriesSize);
      newSeriesID = torch.rand(seriesMissing):mul(curNbSeries):floor():add(1);
      newLabels = trainData.labels.index(trainData.labels, 1, newSeriesID:long());
      collectgarbage();
      for s = 1,seriesMissing do
        if (s % 1000) == 0 then
          collectgarbage();
          print(s);
        end
        local tmpSeries = trainData.data[newSeriesID[s]]:clone();
        local keptSeries = tmpSeries;
        curDensification = math.floor(math.random() * 4);
        if (curDensification == 0) then
          -- Add white gaussian noise
          tmpSeries = tmpSeries:add((torch.rand(seriesSize):add(-0.5):mul(0.1)));
        elseif (curDensification == 1) then
          -- Add random outliers (here similar to dropout mask)
          tmpIDs = torch.rand(math.floor(seriesSize / 20)):mul(seriesSize):floor():add(1);
          tmpSeries[{{tmpIDs}}] = 0 --tmpSeries[{{tmpIDs}}];
        elseif (curDensification == 2) then
          -- Add cropped sub-sequences (uniform warping) 
          local maxCrop = seriesSize * 0.1;
          local cropLeft = math.ceil(math.random() * maxCrop);
          local cropRight = seriesSize - math.ceil(math.random() * maxCrop);
          tmpSeries = tensorResampling(tmpSeries[{{cropLeft,cropRight}}], seriesSize);
        elseif (curDensification == 3) then
          -- Add (non-linear) temporal warped series
          local switchPosition = math.floor(math.random() * (seriesSize / 4)) + (seriesSize / 2);
          leftSide = tmpSeries[{{1,switchPosition}}];
          rightSide = tmpSeries[{{switchPosition+1,seriesSize}}];
          direction = math.random() - 0.5 > 0 and 1 or -1;
          leftSide = tensorResampling(leftSide, switchPosition - (direction * 5));
          rightSide = tensorResampling(rightSide, seriesSize - switchPosition + (direction * 5)); 
          tmpSeries = torch.cat(leftSide, rightSide, 2);
        end
        newSeries[s] = tmpSeries;
      end
      sets[v]["TRAIN"].data = torch.cat(sets[v]["TRAIN"].data, newSeries, 1);
      sets[v]["TRAIN"].labels = torch.cat(sets[v]["TRAIN"].labels, newLabels, 1);
      sets[v]["TRAIN"].mean = torch.cat(sets[v]["TRAIN"].mean, trainData.mean.index(trainData.mean, 1, newSeriesID:long()));
      sets[v]["TRAIN"].std = torch.cat(sets[v]["TRAIN"].std, trainData.std.index(trainData.std, 1, newSeriesID:long()));
    end
  end
  collectgarbage();
  return sets;
end

----------------------------------------------------------------------
-- Constructing a domain-wise unsupervised dataset
----------------------------------------------------------------------
-- function construct_domain(sets, domains)
  -- For each dataset
  -- for v, k in pairs(sets) do
    -- Take only the train set
    -- local trainData = sets[v]["TRAIN"];

----------------------------------------------------------------------
-- Full datasets construction function
----------------------------------------------------------------------

-- Load, organize and optionally preprocess the UCR dataset
function M.import_full(sets, setList, options)
   local setList = setList or {'all'}
   
   print " - Checking data statistics";
   for _, set in ipairs(setList) do
      for _, genericSubset in ipairs({'TRAIN', 'TEST'}) do
	 v = sets[set][genericSubset];
	 meanData = v.data[{{},{}}]:mean();
	 stdData = v.data[{{},{}}]:std();
	 print('    - '..set..' [' .. genericSubset ..'] - '..
		  'mean: ' .. meanData .. ', standard deviation: ' .. stdData);
      end
   end
   if options.dataAugmentation then
      print " - Performing data augmentation";
      --sets = data_augmentation(sets);
   end
   ----------------------------------------------------------------------
   -- Validation and unsupervised datasets
   ----------------------------------------------------------------------

   print " - Constructing balanced validation subsets"
   -- We shall construct a balanced train/validation subset
   sets = construct_validation(sets, options.validPercent);
   print " - Constructing unsupervised superset"
   -- Also prepare a very large unsupervised dataset
   unSets = construct_unsupervised(sets, options.validPercent);

   ----------------------------------------------------------------------
   -- Additional pre-processing code
   ----------------------------------------------------------------------

   local genericSubsets = {"TRAIN", "VALID", "TEST"}
   -- Global Contrast Normalization
   if options.gcnNormalize then
      print ' - Perform Global Contrast Normalization (GCN) on input data'
      unSets["TRAIN"].data = preprocess.gcn(unSets["TRAIN"].data);
      unSets["VALID"].data = preprocess.gcn(unSets["VALID"].data);

      for _, v in ipairs(setList) do
	 for _, genericSubset in ipairs(genericSubsets) do
	    sets[v][genericSubset].data = preprocess.gcn(
	       sets[v][genericSubset].data);
	 end
      end
   end
   -- Zero-Component Analysis whitening
   if options.zcaWhitening then
      print ' - Perform Zero-Component Analysis (ZCA) whitening'
      local means, P = zca_whiten_fit(
	 torch.cat(unSets["TRAIN"].data, unSets["VALID"].data, 1));
      unSets["TRAIN"].data = preprocess.zca_whiten_apply(unSets["TRAIN"].data,
							 means, P)
      unSets["VALID"].data = preprocess.zca_whiten_apply(unSets["VALID"].data,
							 means, P)

      for _, v in pairs(setList) do
	 local means, P = preprocess.zca_whiten_fit(
	    torch.cat(sets[v]["TRAIN"].data, sets[v]["VALID"].data, 1));
	 
	 for _, genericSubset in ipairs(genericSubsets) do
	    sets[v][genericSubset].data = preprocess.zca_whiten_apply(
	       sets[v][genericSubset].data, means, P)
	 end
      end
   end

   return sets, unSets
end

function M.dataset_loader(dataset, options)
   if not type(dataset) == 'string' then error("Must input chosen dataset") end

   local local_importer = function(baseDir, setList, set_import_f)
      print " - Importing datasets";
      local sets = set_import_f(baseDir, setList, options.resampleVal);
      
      return import_full(sets, setList, options)
   end
   if dataset == 'UCR' then
      local baseDir = ucr.baseDir
      local setList = ucr.setList
      
      return local_importer(baseDir, setList, import_ucr_data)
   elseif dataset == 'million_song_subset' then
      local baseDir = msds.baseDir
      local setList = msds.setList
      
      local sets = import_msds_data(baseDir, setList,
				    options.resampleVal);
   end
end

----------------------------------------------------------------------
-- Dataset specific import functions
----------------------------------------------------------------------
function M.import_ucr_data(dirData, setFiles, resampleVal)
  -- Sets data
  local sets = {};
  -- Types of datasets
  local setsTypes = {"TEST", "TRAIN"};
  -- Load the datasets (factored)
  for id,value in ipairs(setFiles) do
    sets[value] = {};
    for idT,valType in ipairs(setsTypes) do
      print("    - Loading " .. value .. " [" .. valType .. "]");
      -- Get the test and train sets
      local trainName = dirData .. "/" .. value .. "/" .. value .. "_" .. valType; 

      -- Parse data-file
      local finalData = ucr.parse(trainName)
      
      if (resampleVal) then
        finalData = tensorResampling(finalData, resampleVal);
      end
      
      -- Make structure
      sets[value][valType] = make_structure(final_data);
    end
    if (collectgarbage("count") > 1000000) then
      print("Collecting garbage for ".. (collectgarbage("count")) .. "Ko");
      collectgarbage();
    end
  end
  -- Just make sure to remove unwanted memory
  collectgarbage();
  return sets;
end

-- Extract a subrange from a table 
local function subrange(t, first, last)
   local sub = {}
   for i=first,last do
      table.insert(sub, t[i])
   end
   return sub
end

-- Sum all elements in a table
local sum = function(elems)
   local plus = function(x, y) return x + y end
   return moses.reduce(elems, plus)
end

-- Deep table copy
function deepcopy(orig)
   local orig_type = type(orig)
   local copy
   if orig_type == 'table' then
      copy = {}
      for orig_key, orig_value in next, orig, nil do
	 copy[deepcopy(orig_key)] = deepcopy(orig_value)
      end
      setmetatable(copy, deepcopy(getmetatable(orig)))
   else -- number, string, boolean, etc
      copy = orig
   end
   return copy
end

-- Return filenames for training, validation and testing sets in chosen dataset
--
-- Return:
--  * sets, a three-elements table sets with sets['TRAIN'], sets['VALID'] and
--  sets['TEST'] filled with filenames
--  * folders, a table holding the folders used for monitoring
-- 
-- Input:
--  * dirData, a string: the location of the full dataset
--  * sets_subfolders, a table of string arrays: the paths to each of the
--   subsets at stake (training, validation and testing)
--  * filter_suffix, a string, optional: the suffix of files to keep
--  * trimPath, a boolean: if set, return filenames in the form
--   {path1 : {filenames}, path2 : {filenames}, ...}, which reduces redundancy
--   in the path names.
function M.import_sets_filenames(dirData, sets_subfolders, filter_suffix, trimPath)
   local filter_suffix = filter_suffix or ''
   
   local sets = {}
   local folders = {}
   for subsetType, paths in pairs(sets_subfolders) do
      sets[subsetType] = {}
      folders[subsetType] = {}
      for _, path in ipairs(paths) do
	 if trimPath then
	    -- Store filenames in a dictionary with foldername as key
	    -- to avoid redundant storing of the foldername
	    sets[subsetType][dirData .. path] = {}
	 end

	 table.insert(folders[subsetType], dirData .. path)
	 
	 local filenames_iterator = diriter.dirtree(dirData .. path, trimPath)
	 local filenames_table = diriter.to_array(filter_suffix, filenames_iterator)
	 for _, filename in ipairs(filenames_table) do
	    if trimPath then
	       table.insert(sets[subsetType][dirData .. path], filename)
	    else
	       table.insert(sets[subsetType], filename)
	    end
	 end
      end
   end
   return sets, folders
end

-- Iterator factory for a sliding window over a dataset
-- 
-- Return a new tensor of sequence slices of uniform duration, using a sliding
-- window of the full dataset.
-- Also return the index of the last loaded file in the training dataset
-- (allows tracking training progress).
-- 
-- Each iteration yieds training, validation (and optionnaly) testing subsets
-- from the dataset.
-- Input:
--  * sets, a table of arrays: the dataset over which to iterate
--  * main_window_size, an integer: the size of the window for the training
--   subsets.
--  * sliding_step, an integer: the amount by which to slide the windows
--   at each iteration.
--  * slice_size, an integer: the duration with which to split all the
--   sequences.
--  * f_load, a function : filename -> torch.Tensor: load a file
function M.get_sliding_window_iterator(filenames, windowSize, windowStep, f_load,
				       options, verbose)
   -- Shuffle the dataset
   local function shuffleTable(tableIn)
      return sampleFile.get_random_subset(tableIn, #tableIn)
   end
   local filenames = shuffleTable(filenames)

   -- Slice all examples in the batch to have same duration,
   -- allows putting them all in a single tensor for memory efficiency
   local slices = nil
   -- Used to store number of slices per example, enabling better memory
   -- management (can update slices variable in place)
   local slicesNumber = nil

   local prev_start, prev_end
   
   local function update_slices(cur_start, cur_end, new_slices, new_slicesNumber,
				prev_position)
      if not(slices) then  -- Initialize containers
	 slicesNumber = new_slicesNumber
	 slices = new_slices
      else
	 slicesNumber = slicesNumber:cat(
	       new_slicesNumber)
	    
	 -- Trim and extend slices container
	 -- Get informations for trimming
	 slicesNumbers_erase = subrange(slicesNumber,
					prev_start,
					cur_start-1)
	 
	 local erase_slices_num_total = sum(slicesNumbers_erase)
	 
	 for dataType, slicesSubsetData in pairs(slices) do
	    -- Iterate over original examples and targets
	    
	    -- Trim left out slices --
	    -- Compute number of slices to drop
	    local current_slices_num = slices[dataType]:size(2)
	    if erase_slices_num_total == current_slices_num then
	       -- Erase all slices
	       slices[dataType] = nil
	    else
	       assert(erase_slices_num_total <= current_slices_num,
		      "Number of slices to delete shouldn't be higher than " ..
			 "current number of slices")
	       slices[dataType] = slices[dataType]:
		  narrow(options.batchDim, 1+erase_slices_num_total,
			 slices[dataType]:size(options.batchDim)-
			    erase_slices_num_total)
	    end
	    collectgarbage()
	    
	    -- Add new slices
	    if slices[dataType] then
	       slices[dataType] = slices[dataType]:cat(
		  new_slices[dataType], options.batchDim)
	    else  -- all slices were erased, initialize new slices 
	       slices[dataType] = new_slices[dataType]
	    end
	 end
      end
   end

   local function f(filenames_window, cur_start, cur_end)
      collectgarbage(); collectgarbage()
      
      -- slice all examples in the batch to have same duration,
      -- allows putting them all in a single tensor for memory efficiency
      local new_slices, new_slicesNumber = M.load_slice_filenames_tensor(
	 filenames_window, f_load, options)

      -- update internal memory
      update_slices(cur_start, cur_end, new_slices, new_slicesNumber, prev_position)
      
      prev_start = cur_start

      return slices
   end

   -- Do not reload pre-loaded files
   local no_overlap = true
   return iterator.foreach_windowed(f, filenames, windowSize, windowStep,
				    no_overlap, verbose)
end

-- -- Iterator factory for a sliding window over a dataset
-- -- 
-- -- Return a new tensor of sequence slices of uniform duration, using a sliding
-- -- window of the full dataset.
-- -- Also return the index of the last loaded file in the training dataset
-- -- (allows tracking training progress).
-- -- 
-- -- Each iteration yieds training, validation (and optionnaly) testing subsets
-- -- from the dataset.
-- -- Input:
-- --  * sets, a table of arrays: the dataset over which to iterate
-- --  * main_window_size, an integer: the size of the window for the training
-- --   subsets.
-- --  * sliding_step, an integer: the amount by which to slide the windows
-- --   at each iteration.
-- --  * slice_size, an integer: the duration with which to split all the
-- --   sequences.
-- --  * f_load, a function : filename -> torch.Tensor: load a file
-- function M.__get_sliding_window_iterator(filenames, f_load, options,
-- 					 no_overlap)
--    local filenames_num = #filenames
--    local window_size = math.min(options.datasetWindowSize, filenames_num)
--    local sliding_step = options.datasetWindowStepSize or
--       math.floor(window_size / 10)

--    -- Use this for a non-overlapping sliding window (last step then uses
--    -- a smaller window)
--    local no_overlap = no_overlap or false

--    local slice_size = options.sliceSize
   
--    -- Avoid skipping some examples because of a step too long
--    if sliding_step > window_size then
--       sliding_step = window_size
--    end
   
--    local function euclidean_division(a, b)
--       local quotient = math.floor(a / b)
--       local remainder = a % b
      
--       return quotient, remainder
--    end
   
--    local steps_num, remainder_step = euclidean_division(
--       filenames_num - window_size, sliding_step)
   
--    print(window_size) -- DEBUG
--    print(steps_num) -- DEBUG
--    local sizes = {}
--    sizes['winSize'] = window_size
--    sizes['stepSize'] = sliding_step
--    sizes['lastStep'] = remainder_step
   
--    -- Shuffle the dataset
--    local function shuffleTable(tableIn)
--       return sampleFile.get_random_subset(tableIn, #tableIn)
--    end
--    local filenames = shuffleTable(filenames)
--    print(filenames)

--    -- current position of window on the dataset
--    local position = 1

--    -- Slice all examples in the batch to have same duration,
--    -- allows putting them all in a single tensor for memory efficiency
--    local slices
--    -- Used to store number of slices per example, enabling better memory
--    -- management (can update slices variable in place)
--    local slicesNumber
   
--    local function get_window(step_n)
--       if step_n > steps_num then return nil end
      
--       -- If window size larger than number of files, return whole set of filenames
--       if window_size >= filenames_num then return filenames end
      
--       local filenames_window = {}
      
--       local win_size = sizes['winSize']
--       local step_size = sizes['stepSize']
      
--       -- get length of next step
--       local next_step
--       if step_n < steps_num then
-- 	 next_step = step_size  -- a normal step
--       else
-- 	 -- in that case, step_n == steps_num, make smaller step
-- 	 -- with the remaining files
-- 	 if no_overlap then
-- 	    next_step = step_size
-- 	    win_size = sizes['lastStep']
-- 	 else
-- 	    next_step = sizes['lastStep']
-- 	 end
--       end
      
--       local win_start = (step_n-1) * step_size + next_step + 1
--       -- Update positions in dataset
--       position = win_start
--       local win_end = win_start + win_size - 1
      
--       if step_n > 0 then
-- 	 -- Only return new files within the sliding window
-- 	 -- (i.e. end of full window)
-- 	 win_start = win_start + (win_size - next_step)
--       end
      
--       if win_end == filenames_num then print('YAY!') end      

--       for file_n = win_start, win_end do
-- 	 filenames_window[1 + file_n-win_start] = filenames[file_n]
--       end
      
--       return filenames_window
--    end
   
--    local function update_slices(step_n, new_slices, new_slicesNumber,
-- 				prev_position)
--       if step_n == 0 then  -- Initialize containers
-- 	 slicesNumber = new_slicesNumber
-- 	 slices = new_slices
--       elseif step_n < steps_num then
-- 	 slicesNumber = slicesNumber:cat(
-- 	       new_slicesNumber)
	    
-- 	 -- Trim and extend slices container
-- 	 -- Get informations for trimming
-- 	 prev_win_start = prev_position
-- 	 step_size = sizes['stepSize']
-- 	 slicesNumbers_erase = subrange(slicesNumber,
-- 					prev_win_start,
-- 					prev_win_start+step_size-1)
	 
-- 	 local erase_slices_num_total = sum(slicesNumbers_erase)
	 
-- 	 for dataType, slicesSubsetData in pairs(slices) do
-- 	    -- Iterate over original examples and targets
	    
-- 	    -- Trim left out slices --
-- 	    -- Compute number of slices to drop
-- 	    local current_slices_num = slices[dataType]:size(2)
-- 	    if erase_slices_num_total == current_slices_num then
-- 	       -- Erase all slices
-- 	       slices[dataType] = nil
-- 	    else
-- 	       assert(erase_slices_num_total <= current_slices_num,
-- 		      "Number of slices to delete shouldn't be higher than " ..
-- 			 "current number of slices")
-- 	       slices[dataType] = slices[dataType]:
-- 		  narrow(options.batchDim, 1+erase_slices_num_total,
-- 			 slices[dataType]:size(options.batchDim)-
-- 			    erase_slices_num_total)
-- 	    end
-- 	    collectgarbage()
	    
-- 	    -- Add new slices
-- 	    if slices[dataType] then
-- 	       slices[dataType] = slices[dataType]:cat(
-- 		  new_slices[dataType], options.batchDim)
-- 	    else  -- all slices were erased, initialize new slices 
-- 	       slices[dataType] = new_slices[dataType]
-- 	    end
-- 	 end
--       end
--    end
   
--    local step_n = 0
--    local file_position = 1
   
--    return function()
--       collectgarbage(); collectgarbage()
--       if step_n > steps_num then return nil end
      
--       -- copy previous position
--       local prev_position = position
--       -- get new files to load
--       local filenames_window = get_window(step_n)

--       file_position = file_position + #filenames_window
      
--       -- slice all examples in the batch to have same duration,
--       -- allows putting them all in a single tensor for memory efficiency
--       local new_slices, new_slicesNumber = M.load_slice_filenames_tensor(
-- 	 filenames_window, f_load, options)

--       -- update internal memory
--       update_slices(step_n, new_slices, new_slicesNumber, prev_position)

--       step_n = step_n + 1
      
--       return slices, file_position
--    end
-- end

-- Load various subsets of filenames into tensors
-- 
-- Also return slicesNumber, a table containing for each subset the number
-- of slices yielded by the files in it, in order, thus allowing to properly
-- trim slices tensors afterwards.
--
-- TODO: the way targets are stored is suboptimal, replicating most of the
-- contents of the data. Could optimize by only returning the actual new steps
-- to predict. 
-- 
-- Input:
--  * sets, a table of string tables: the training, validation... sets,
--   as filenames
--  * f_load, a function : string -> sequence: sequence loading function
--  * options, a table, with the following fields:
--    * options.sliceSize, an integer, optional: the size by which to slice
--     the sequences (default 128)
--    * options.predictionLength, an integer: the duration over which to do
--     the forward prediction (default 1 step, predict the next item in the
--     sequence)
function M.load_slice_sets_tensor(sets, f_load, options)
   local slicedData = {}
   local slicesNumbers = {}
   
   for subsetType, subset_filenames in pairs(sets) do
      print('Loading ' .. subsetType .. ' dataset')
      
      slicedData[subsetType] = {}

      slicedData_subset, slicesNumbers_subset = M.load_slice_filenames_tensor(
	 subset_filenames, f_load, options)
      
      slicedData[subsetType] = slicedData_subset 
      slicesNumbers[subsetType] = slicesNumbers_subset
   end
   
   return slicedData, slicesNumbers
end

-- Take a set of sequences as filenames and return a tensor of equal-size slices
-- (Specialized function for M.load_slice_sets_tensor())
-- 
-- Input:
--  * sets, a table of string tables: the training, validation... sets,
--   as filenames
--  * f_load, a function : string -> sequence: sequence loading function
--  * options, a table, with the following fields:
--    * options.sliceSize, an integer, optional: the size by which to slice
--     the sequences (default 128)
--    * options.predictionLength, an integer: the duration over which to do
--     the forward prediction (default 1 step, predict the next item in the
--     sequence)
--    * options.paddingValue, a float: the value to add in front of sequences
--     too short to be sliced
--  * folderName, a string, optional: a root path for the given filenames
--     (default '') 
function M.load_slice_filenames_tensor(filenames, f_load, options, folderName)
   if not(filenames) or next(filenames) == nil then return {}, torch.Tensor() end

   local paddingValue = options.paddingValue or 0
   
   local folderName = folderName or ''
   local function makeFullFilename(filename)
      return folderName .. filename
   end

   local sliceSize = options.sliceSize or 128
   local predictionLength = options.predictionLength or 1

   assert(predictionLength < sliceSize,
	  "Can't predict more than length of the slices")
   
   -- Initialize containers
   local example = f_load(makeFullFilename(filenames[1]))  -- get an example in the batch
   local featSize = example:size(2)  -- dimension of the time-series
   local slicedData = torch.zeros(sliceSize, 1, featSize)
   local slicedTargets
   if options.predict then
      slicedTargets = torch.zeros(predictionLength, 1, featSize)
   else
      slicedTargets = torch.zeros(sliceSize, 1, featSize)
   end
   local slicesNumbers = torch.zeros(#filenames)
   
   -- Slice input sequence into equal sized windows
   -- 
   -- Return: a tensor of slices with dimension sliceSize x slicesNumber x featSize
   local slicer = nn.SequencerSlidingWindow(1, sliceSize, sliceSize)
   local function sliceSequence(sequence)
      local sequenceDuration = sequence:size(1)
      
      sequence = sequence:view(sequenceDuration, 1, featSize)
      local slices = slicer:forward(sequence)
      return slices
   end
   
   -- Adjust duration of sequences to properly extract slices
   -- Padding is added from the last: sequence starts from void
   local function pad(sequence)
      local sequenceDuration = sequence:size(1)
      
      if sequenceDuration < sliceSize + predictionLength then
	 -- Sequence is shorter than the slice size: add silence at the beginning
	 local deltaDuration = sliceSize - sequenceDuration

	 local padding = torch.Tensor(deltaDuration + predictionLength, featSize)
	 padding:fill(paddingValue)
	 
	 local paddedSequence = padding:cat(sequence, 1)
	 return paddedSequence
      else
	 return sequence
      end
   end

   -- Extract only actual predictions from a batch of examples
   -- Sequence loader returns targets as simply offset versions of their data
   -- counterpart, this extracts the predictionLength tail of those sequences
   local function selectPrediction(batch, options)
      local sliceSize = batch:size(options.tDim)
      return batch:narrow(1, sliceSize-options.predictionLength+1, options.predictionLength)
   end
   
   -- for subsetType, subset in pairs(sets) do
   for sequence_i, filename in ipairs(filenames) do
      if sequence_i % 300 == 0 then
	 -- print(slicedData:size())
	 collectgarbage(); collectgarbage()
      end

      -- print('Files: ' .. sequence_i .. ', memory usage: ' .. collectgarbage('count'))
      
      -- Load the sequences step by step to avoid crashing Lua's memory with a table
      local fullFilename = makeFullFilename(filename)
      local sequence = f_load(fullFilename)
      local sequence = pad(sequence)
      local sequenceDuration = sequence:size(1)
      
      local slices = sliceSequence(sequence)
      slicesNumbers[sequence_i] = slices:size(options.batchDim)
      
      -- Pad end of sequence with silence to compensate for prediction offset
      local offsetPadding = torch.Tensor(predictionLength, featSize)
      offsetPadding:fill(paddingValue)
      local offsetSequence = sequence:narrow(
	 1, 1+predictionLength, sequenceDuration-predictionLength):cat(
	 offsetPadding, 1)
      local targetSlices = sliceSequence(offsetSequence)
      
      if options.predict then
	 targetSlices = selectPrediction(targetSlices, options)
      end
      
      slicedData = slicedData:cat(slices, options.batchDim)
      slicedTargets = slicedTargets:cat(targetSlices, options.batchDim)
   end
   
   -- Remove initial zeros slice
   local slicesNumber = slicedData:size(options.batchDim)
   slicedData = slicedData:narrow(options.batchDim, 2, slicesNumber-1)
   slicedTargets = slicedTargets:narrow(options.batchDim, 2, slicesNumber-1)
   
   collectgarbage(); collectgarbage()

   slicedData_table = {data = slicedData,
		       targets = slicedTargets}
   
   return slicedData_table, slicesNumbers
end

-- Use this function if using filenames sets with the format
-- {path1 : {filenames}, path2 : {filenames}, ...} 
function M.load_slice_filenamesFolders_tensor(filenamesFolders, f_load, options)
   -- Get folders list
   local function foldersList(filenamesFolders)
      local folders = {}
      for folder, _ in pairs(filenamesFolders) do
	 table.insert(folders, folder)
      end
      return folders
   end

   local folders = foldersList(filenamesFolders)

   local slicedData_table, slicesNumbers = M.load_slice_filenames_tensor(
      filenamesFolders[folders[1]], f_load, options, folders[1])

   for folder_n=2,#folders do
      local foldername = folders[folder_n]
      print(foldername)
      local new_slicedData_table, new_slicesNumbers = M.load_slice_filenames_tensor(
      filenamesFolders[foldername], f_load, options, foldername)

      for dataType, _ in pairs(slicedData_table) do
	 slicedData_table[dataType] = slicedData_table[dataType]:cat(
	    new_slicedData_table[dataType], options.batchDim)
      end
      slicesNumbers = slicesNumbers:cat(new_slicesNumbers)
   end

   return slicedData_table, slicesNumbers
end


return M
