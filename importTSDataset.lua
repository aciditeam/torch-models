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
-- Return: a three-elements table sets with sets['TRAIN'], sets['VALID'] and
--  sets['TEST'] filled with filenames
-- Input:
--  * dirData, a string: the location of the full dataset
--  * sets_subfolders, a table of string arrays: the paths to each of the
--   subsets at stake (training, validation and testing)
--  * filter_suffix, a string, optional: the suffix of files to keep
function M.import_sets_filenames(dirData, sets_subfolders, filter_suffix)
   local filter_suffix = filter_suffix or ''
   
   local sets = {}
   for subsetType, paths in pairs(sets_subfolders) do
      sets[subsetType] = {}
      for _, path in ipairs(paths) do
	 filenames_iterator = diriter.dirtree(dirData .. path)
	 filenames_table = diriter.to_array(filter_suffix, filenames_iterator)
	 for _, filename in ipairs(filenames_table) do
	    table.insert(sets[subsetType], filename)
	 end
      end
   end
   return sets
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
function M.get_sliding_window_iterator(sets, f_load, options)
   local main_window_size = options.datasetWindowSize
   local sliding_step = options.datasetWindowStepSize or
      math.floor(main_window_size / 10)

   local slice_size = options.sliceSize

   local train_examples_num = #sets['TRAIN']
   main_window_size = math.min(main_window_size, train_examples_num)
   
   -- Avoid skipping some examples because of a step too long
   if sliding_step > main_window_size then
      sliding_step = main_window_size
   end
   
   local function euclidean_division(a, b)
      local quotient = math.floor(a / b)
      local remainder = a % b
      
      return quotient, remainder
   end
   
   local steps_num, remainder_step = euclidean_division(
      train_examples_num - main_window_size, sliding_step)
   
   -- Shuffle the datasets
   local shuffled_sets = {}
   for subsetType, subset in pairs(sets) do
      local shuffle = sampleFile.get_random_subset(subset, #subset)
      shuffled_sets[subsetType] = shuffle
   end
   
   -- Compute size of windows for auxiliary subsets (validation, testing...)
   -- Sizes are chosen to be of the same ratio as for the training set.
   -- 
   -- EDIT: Do not use this, instead use a fixed validation set!
   local function get_win_step_sizes(elems, subsetType, sizes)
      local elems_num = #elems
      local min_win_size = math.min(elems_num, 1)
      local win_size = math.max(
	 math.floor(elems_num * (main_window_size/train_examples_num)),
	 min_win_size)
      local step_size, last_step = euclidean_division(elems_num - win_size,
						      steps_num)
      sizes[subsetType] = {}
      sizes[subsetType]['winSize'] = win_size
      sizes[subsetType]['stepSize'] = step_size
      sizes[subsetType]['lastStep'] = last_step
   end
   
   -- Initialize window sizes and current positions for each subsets
   local sizes = {}  -- size of window for each subset of the dataset
   local positions = {}  -- current position of window on the datasets
   for subsetType, subset in pairs(sets) do
      get_win_step_sizes(subset, subsetType, sizes)
      positions[subsetType] = 1
   end

   -- Slice all examples in the batch to have same duration,
   -- allows putting them all in a single tensor for memory efficiency
   local slices
   -- Used to store number of slices per example, enabling better memory
   -- management (can update slices variable in place)
   local slicesNumber
   
   local function get_windows(step_n)
      if step_n > steps_num then return nil end
      
      local windowed_sets = {}
      for subsetType, subset in pairs(sets) do
	 windowed_sets[subsetType] = {}
	 
	 local win_size = sizes[subsetType]['winSize']
	 local step_size = sizes[subsetType]['stepSize']
	 
	 -- get length of next step
	 local next_step
	 if step_n < steps_num then
	    next_step = step_size  -- a normal step
	 else
	    -- in that case, step_n == steps_num, make smaller step
	    -- with the remaining files
	    next_step = sizes[subsetType]['lastStep']
	 end
	 
	 local win_start = (step_n-1) * step_size + next_step + 1
	 -- Update positions in dataset
	 positions[subsetType] = win_start
	 local win_end = win_start + win_size - 1
	 
	 if step_n > 0 then
	    -- Only return new files within the sliding window
	    -- (i.e. end of full window)
	    win_start = win_start + (win_size - next_step)
	 end
	 print('New files window: {start = ' ..win_start ..
	       ', end = ' .. win_end .. '}')
	 
	 for file_n = win_start, win_end do
	    windowed_sets[subsetType][1 + file_n-win_start] = subset[file_n]
	 end
      end
      
      return windowed_sets
   end
   
   local function update_slices(step_n, new_slices, new_slicesNumber,
				prev_positions)
      if step_n == 0 then  -- Initialize containers
	 slicesNumber = new_slicesNumber
	 slices = new_slices
      elseif step_n < steps_num then
	 for subsetType, slicesSubset in pairs(new_slices) do
	    slicesNumber[subsetType] = slicesNumber[subsetType]:cat(
	       new_slicesNumber[subsetType])
	    
	    -- Trim and extend slices container
	    -- Get informations for trimming
	    prev_win_start = prev_positions[subsetType]
	    step_size = sizes[subsetType]['stepSize']
	    slicesNumbers_erase = subrange(slicesNumber[subsetType],
					   prev_win_start,
					   prev_win_start+step_size-1)
	    
	    local erase_slices_num_total = sum(slicesNumbers_erase)
	    
	    for dataType, slicesSubsetData in pairs(slicesSubset) do
	       -- Iterate over original examples and targets
	       
	       -- Trim left out slices --
	       
	       -- Compute number of slices to drop
	       local current_slices_num = slices[subsetType][dataType]:size(2)
	       if erase_slices_num_total == current_slices_num then
		  -- Erase all slices
		  slices[subsetType][dataType] = nil
	       else
		  assert(erase_slices_num_total < current_slices_num,
			 "Number of slices to delete shouldn't be higher than " ..
			    "current number of slices")
		  slices[subsetType][dataType] = slices[subsetType][dataType]:
		     narrow(options.batchDim, 1+erase_slices_num_total,
			    slices[subsetType][dataType]:size(options.batchDim)-
			       erase_slices_num_total)
	       end
	       collectgarbage()
	       
	       -- Add new slices
	       if slices[subsetType][dataType] then
		  slices[subsetType][dataType] = slices[subsetType][dataType]:cat(
		     new_slices[subsetType][dataType], options.batchDim)
	       else  -- all slices were erased, initialize new slices 
		  slices[subsetType][dataType] = new_slices[subsetType][dataType]
	       end
	    end
	 end
      end
   end
   
   local step_n = 0
   local file_position = 0
   
   return function()
      collectgarbage(); collectgarbage()
      if step_n == steps_num then return nil end
      
      -- copy previous positions
      local prev_positions = deepcopy(positions)
      -- get new files to load
      local windowed_sets = get_windows(step_n)

      file_position = file_position + #windowed_sets['TRAIN']
      
      -- slice all examples in the batch to have same duration,
      -- allows putting them all in a single tensor for memory efficiency
      local new_slices, new_slicesNumber = M.load_slice_sets_tensor(
	 windowed_sets, f_load, options)

      -- update internal memory
      update_slices(step_n, new_slices, new_slicesNumber, prev_positions)

      step_n = step_n + 1
      
      return slices, file_position
   end
end

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

      slicedData_subset, slicedTargets_subset, slicesNumbers_subset = M.load_slice_filenames_tensor(
	 subset_filenames, f_load, options)
      
      slicedData[subsetType]['data'] = slicedData_subset 
      slicedData[subsetType]['targets'] = slicedTargets_subset
      slicesNumbers[subsetType] = slicesNumbers_subset
   end
   
   return slicedData, slicesNumbers
end

-- Take a set of sequences as filenames and return a tensor of equal-size slices
-- (Specialized function for M.load_slice_sets_tensor()) 
function M.load_slice_filenames_tensor(filenames, f_load, options)
   if not(filenames) or filenames == {} then return {} end
   
   local sliceSize = options.sliceSize or 128
   local predictionLength = options.predictionLength or 1
   
   -- Initialize containers
   local example = f_load(filenames[1])  -- get an example in the batch
   local featSize = example:size(2)  -- dimension of the time-series
   local slicedData = torch.zeros(sliceSize, 1, featSize)
   local slicedTargets = torch.zeros(sliceSize, 1, featSize)
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
   local function zeroPad(sequence)
      local sequenceDuration = sequence:size(1)
      
      if sequenceDuration < sliceSize + predictionLength then
	 -- Sequence is shorter than the slice size: add silence at the end
	 local deltaDuration = sliceSize - sequenceDuration
	 local zeroPaddedSequence = sequence:cat(
	    torch.zeros(deltaDuration + predictionLength, featSize), 1)
	 return zeroPaddedSequence
      else
	 return sequence
      end
   end
   
   -- for subsetType, subset in pairs(sets) do
   for sequence_i, filename in ipairs(filenames) do
      -- print('Files: ' .. sequence_i .. ', memory usage: ' .. collectgarbage('count'))
      
      -- Load the sequences step by step to avoid crashing Lua's memory with a table
      local sequence = f_load(filename)
      local sequence = zeroPad(sequence)
      local sequenceDuration = sequence:size(1)
      
      local slices = sliceSequence(sequence)
      slicesNumbers[sequence_i] = slices:size(options.batchDim)
      
      -- TODO: can maybe improve this, could replace added zeros (necessary
      -- to ensure same number of slices for original sequence and targets)
      -- by values from the actual original sequence
      local offsetSequence = sequence:narrow(1, 1+predictionLength,
					     sequenceDuration-predictionLength):
	 cat(torch.zeros(predictionLength, featSize), 1)
      local targetSlices = sliceSequence(offsetSequence)
      
      slicedData = slicedData:cat(slices, options.batchDim)
      slicedTargets = slicedTargets:cat(targetSlices, options.batchDim)
   end
   
   -- Remove initial zeros slice
   local slicesNumber = slicedData:size(options.batchDim)
   slicedData = slicedData:narrow(options.batchDim, 2, slicesNumber-1)
   slicedTargets = slicedTargets:narrow(options.batchDim, 2, slicesNumber-1)
   
   collectgarbage(); collectgarbage()
   
   print('Loaded ' .. slicedData:size(options.batchDim) .. ' new slices')
   
   return slicedData, slicedTargets, slicesNumbers
end

return M
