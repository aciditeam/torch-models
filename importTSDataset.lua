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
-- Each iteration yieds training, validation (and optionnaly) testing subsets
-- from the dataset.
-- Input:
--  * sets, a table of arrays: the dataset over which to iterate
--  * main_window_size, an integer: the size of the window for the training
--   subsets.
--  * sliding_step, an integer: the amount by which to slide the windows
--   at each iteration.
function M.get_sliding_window_iterator(sets, main_window_size, sliding_step)
   local train_examples_num = #sets['TRAIN']

   local function euclidean_division(a, b)
      local quotient = math.floor(a / b)
      local remainder = a % b
      
      return quotient, remainder
   end

   local steps_num, remainder_step = euclidean_division(
      train_examples_num - main_window_size, sliding_step)
   
   -- local leftover_examples_n = train_examples_num % main_window_size

   -- Will split the dataset in windows_n windows + the remaining examples
   -- local windows_num = (train_examples_num - leftover_examples_n) /
   --    main_window_size

   -- Shuffle the datasets
   local shuffled_sets = {}
   for subsetType, subset in pairs(sets) do
      local shuffle = sampleFile.get_random_subset(subset, #subset)
      shuffled_sets[subsetType] = shuffle
   end

   -- Compute size of windows for auxiliary subsets (validation, testing...)
   -- Sizes are chosen to be of the same ratio as for the training set.
   local function get_win_step_sizes(elems, subsetType, sizes)
      local elems_num = #elems
      local win_size = math.max(
	 math.floor(elems_num * (main_window_size/train_examples_num)), 1)
      local step_size, last_step = euclidean_division(elems_num - win_size,
						      steps_num)
      sizes[subsetType] = {}
      sizes[subsetType]['winSize'] = win_size
      sizes[subsetType]['stepSize'] = step_size
      sizes[subsetType]['lastStep'] = last_step
   end

   -- Initialize window sizes for each subsets
   local sizes = {}
   for subsetType, subset in pairs(sets) do
      get_win_step_sizes(subset, subsetType, sizes)
   end
   
   local function make_windows(elems, step_n)
      if step_n > steps_num then return nil end

      local windowed_sets = {}
      for subsetType, subset in pairs(sets) do
	 windowed_sets[subsetType] = {}
	 
	 win_size = sizes[subsetType]['winSize']
	 step_size = sizes[subsetType]['stepSize']

	 -- Get length of step to do now
	 if step_n < steps_num then
	    next_step = step_size  -- a normal step
	 else  -- step_n == steps_num, smaller leftover step
	    next_step = sizes[subsetType]['lastStep']
	 end
	 local win_start = (step_n-1) * step_size + next_step
	 local win_end = win_start + win_size
	 for file_n = win_start, win_end do
	    windowed_sets[subsetType][file_n] = subset[win_start + file_n + 1]
	 end
      end

      return windowed_sets
   end
   
   local step_n = 0
   
   return function()
      local windowed_sets = make_windows(elems, step_n)
      step_n = step_n + 1
      return windowed_sets
   end
end

-- Take a set of sequences as filenames and return a tensor of equal-size slices
-- 
-- Input:
--  * sets, a table of string tables: the training, validation... sets,
--   as filenames
--  * f_load, a function : string -> sequence: sequence loading function
--  * splitSize, an integer, optional: the size by which to slice the sequences
--   (default 128)
function M.load_sets_tensor(sets, f_load, sliceSize)
   local sliceSize = sliceSize or 128
   local slicer = nn.SlidingWindow(1, sliceSize, sliceSize, 1e9, true)

   
   local function map(f, elems)
	    local f_elems = {}
	    for _, elem in ipairs(elems) do
	       table.insert(f_elems, f(elem))
	    end
	    return f_elems
	 end
   
   slicedData = {}

   for subsetType, subset in pairs(sets) do
      local sequences = map(f_load, subset)
      slicedData[subsetType] = {}
      slicedData[subsetType]['data'] = torch.zeros(1, sliceSize,
						   sequences[1]:size(2))
      for _, sequence in pairs(sequences) do
	 local slices = slicer:forward(sequence)
	 slicedData[subsetType]['data'] = slicedData[subsetType]['data']:cat(
	    slices, 1)
      end
   end
   return slicedData
end
   
-- function import_msds_data(dirData, resampleVal)
--   -- Sets data
--   local sets = {};
--   -- Types of datasets
--   local setsTypes = {"TEST", "TRAIN"};
--   -- Load the datasets (factored)
--   sets['all'] = {};
--   for idT,valType in ipairs(setsTypes) do
--      print("    - Loading " .. value .. " [" .. valType .. "]");
--      -- Get the test and train sets
--      TODO

--      -- Parse data-file
--      local finalData = msds.parse(trainName)
     
--      if (resampleVal) then
--         finalData = tensorResampling(finalData, resampleVal);
--      end
     
--      -- Make structure
--      sets[value][valType] = curData;
--   end
--   if (collectgarbage("count") > 1000000) then
--      print("Collecting garbage for ".. (collectgarbage("count")) .. "Ko");
--      collectgarbage();
--   end
--   -- Just make sure to remove unwanted memory
--   collectgarbage();
--   return sets;
-- end

return M
