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
local ucr = require './importUCR'
local preprocess = require './mainPreprocess'

----------------------------------------------------------------------
-- A real resampling function for time series
----------------------------------------------------------------------
function tensorResampling(data, destSize, type)
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
          interpolator.weight[i][j] = (2 * (math.abs(x) ^ 3)) - (3 * (math.abs(x) ^ 2)) + 1;
        elseif type == 'lanczos' then
          interpolator.weight[i][j] = (2 * (math.abs(x) ^ 3)) - (3 * (math.abs(x) ^ 2)) + 1;
        end
      end
    end
  end
  -- print(interpolator.weight);
  return interpolator:forward(data);
end

----------------------------------------------------------------------
-- Basic import function for UCR time series with normalization (and resampling)
----------------------------------------------------------------------
function import_ucr_data(dirData, setFiles, resampleVal)
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
      sets[value][valType] = curData;
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
function construct_validation(sets, validPercent)
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
function construct_unsupervised(sets, validPercent)
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
function data_augmentation(sets)
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
      for s = 1,seriesMissing do
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
function import_ucr_full(baseDir, setList, options)
   print " - Importing datasets";
   local baseDir = baseDir or ucr.baseDir
   local setList = setList or ucr.setList
   
   local sets = import_ucr_data(baseDir, setList,
				options.resampleVal);
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

function dataset_loader(dataset)
   if not dataset then
      
   
end
