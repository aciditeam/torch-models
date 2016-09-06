----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for prediction
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'unsup'
require 'optim'
require 'torch'
require 'nninit'
require './moduleSlidingWindow'
local moduleSequencerSlidingWindow = require 'SequencerSlidingWindow'
local import_dataset = require './importTSDataset'
require 'mainLearning'
require 'mainParameters'
require 'mainProfiler'
require 'mainUtils'
require 'modelCriterias'
-- Feed-forward models
require 'modelMLP'
require 'modelAE'
require 'modelISTA'
-- Convolutional models
require 'modelCNN'
require 'modelVGG'
require 'modelNIN'
require 'modelInception'
require 'modelResidual'
require 'modelTransformer'
require 'modelZeiler'
-- Recurrent models
require 'modelGRU'
require 'modelRNN'
require 'modelRNN-Bidirectional'
require 'modelRNN-LSTM'
require 'modelLSTM'
require 'modelLSTM-Bidirectional'
require 'modelLSTM-Conv'
require 'modelNTM'
require 'modelRAM'
-- Siamese models
require 'modelSiamese'
require 'modelSiameseProduct'
require 'modelSiameseTriplet'
require 'modelDRLIM'
-- Adversarial models
require 'modelGAN'
require 'modelGANClass'
-- Variational models
require 'modelVAE'
require 'modelVariational'
require 'modelDRAW'
-- Graphical models

-- Custom criterions
require 'criterionAcc'

-- Hyper-parameters
require './hyperSampler'
require './hyperParameters'

local sampleFile = require './datasets/sampleFile'

local _ = require 'moses'

----------------------------------------------------------------------
-- Initialization
----------------------------------------------------------------------

local ts_init = require './TSInitialize'
local locals = require './local'

local saveFolder = locals.paths.timeSeriesResults

----------------------------------------------------------------------
-- Get general options + Optional CUDA usage
----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:option('--useCuda', false, 'whether to enable CUDA processing')
cmd:option('--fastRun', false, 'whether to perform a test on very limited models')
cmd:option('--useSubset', false, 'whether to use only the msds subset')
cmd:option('--smallValidSet', false, 'whether to use a smaller validation set for faster computation')
cmd:option('--smallTrainSet', false, 'whether to use a smaller training set for faster computation')
cmd:option('--manualMode', false, 'whether to disable hyper-optimization and train only on a chosen structure')
cmd:option('--loadModel', '', 'select a previously stored model to further train or test (implies --manualMode)')
cmd:option('--testOnly', false, 'only perform computation of test error, no training' ..
	      '\n(implies --manualMode)')
cmd:option('--modelUniform', false, 'use uniform sampling as prediction model')
cmd:option('--modelRepeat', false, 'use memoryless repeat as prediction model')

-- TODO
-- cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
-- cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
-- cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
-- cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
-- TODO

-- parse input params
cmd_params = cmd:parse(arg)
cmd_params.manualMode = cmd_params.manualMode or cmd_params.loadModel
assert(not(cmd_params.testOnly) or cmd_params.loadModel~='' or
	  cmd_params.modelUniform or cmd_params.modelRepeat, 'Must provide model to test')

local options = ts_init.get_options(cmd_params.useCuda)

local function append(appTable, mainTable)
   for k, v in pairs(appTable) do
      mainTable[k] = appTable[k]
   end
end
append(cmd_params, options)

---------------------------------------
-- Conventions
---------------------------------------
-- RNN-library's batch conventions
options.tDim = 1
options.batchDim = 2
options.featsDim = 3

ts_init.set_globals(); ts_init.set_cuda(options)

---------------------------------------
-- Data loading options
---------------------------------------
-- All sequences will be sliced into sub-sequences of this duration
options.sliceSize = 128

options.paddingValue = 0  -- Pad sequences too short to get a slice with this value
-- 0 is for silence in chromagrams

-- Not all the dataset is loaded into memory in a single pass,
-- we perform a sliding window over it, load a subset, perform
-- some iterations of the optimisation process, then slide the window
options.datasetWindowSize = 500
options.datasetMaxEpochs = 2
options.datasetWindowStep = options.datasetWindowSize -- math.floor(options.datasetWindowSize / 2) 

options.printValidation = false  -- Disable this to discard intermediate validation error computations
options.validationRate = 3

---------------------------------------
-- Training options
---------------------------------------
-- Maximum number of successive times the validation error can increase
-- before stopping training
options.maxValidIncreasedEpochs = 5

-- Training parameters
options.batchSize = 64
options.learningRate = 1e-4

-- Numbers of layers to consider
options.minNbLayers = 3
options.maxNbLayers = 5

options.featsNum = 12 -- validData.data:size(options.featsDim)

---------------------------------------
-- Prediction options
---------------------------------------
options.predict = true
options.predictionLength = 1

---------------------------------------
-- TEST/VALIDATION OPTIONS
---------------------------------------
options.testSlidingWindowSize = 16

---------------------------------------
-- Utilities
---------------------------------------
-- Extract only actual predictions from a batch of examples
-- Sequence loader returns targets as simply offset versions of their data
-- counterpart, this extracts the predictionLength tail of those sequences
local function selectPrediction(batch, options)
   local sliceSize = batch:size(options.tDim)
   return batch:narrow(1, sliceSize-options.predictionLength+1, options.predictionLength)
end

-- Perform prediction test using a sliding window
-- Sliding windows use no overlap here
local function slidingTest(model, criterion, filenames, f_load, options, hyperParams)
   local err = 0
   local filenames_num = #filenames
   
   for slice, cur_start, cur_end in import_dataset.get_sliding_window_iterator(
      filenames, options.datasetWindowSize, options.datasetWindowSize,
      f_load, options) do
      xlua.progress(cur_start, filenames_num)

      err = err + unsupervisedTest(model, criterion, slice, options);
      print('Current cumulative error: ' .. err)
   end
   return err
end

-- Debug and printing parameters
-- Print current validation every ... analyzed files
options.printValidationRate = 5000
-- Return a function periodically printing validation error of model of data
local function getValidationPrinter(model, criterion, validFilenames,
				    f_load, options, hyperParams)
   if options.printValidation then
      local previous_file_position = 1
      local previous_print_valid = 0
      
      local function validationPrinter(file_position)
	 local loaded_files = file_position - previous_file_position
	 previous_print_valid = previous_print_valid + loaded_files
	 
	 if previous_print_valid >= options.printValidationRate then
	    print('\n\tComputing current validation error:')
	    validErr = slidingTest(model, criterion, validFilenames,
				   f_load, options, hyperParams);
	    print('\n\tCurrent validation error: ' .. validErr)
	    previous_print_valid = previous_print_valid %
	       options.printValidationRate
	 end
	 previous_file_position = file_position
      end
      return validationPrinter
   else
      return function() end
   end
end

----------------------------------------------------------------------
-- Initialize dataset
----------------------------------------------------------------------

local msds = require './importMSDS'

local f_load = msds.load.get_btchromas

-- local _, filenamesSets = ts_init.import_data(baseDir, setList, options)
local filter_suffix = '.dat'
if filter_suffix ~= '.dat' then
   print('WARNING: computing chromagrams, prefer usage of .dat files if possible')
end

local datasetPath, datasetSets
local useSubset = false
if options.fastRun or options.useSubset then
   datasetPath = msds.subset.path
   datasetSets = msds.subset.sets
else  -- full dataset
   datasetPath = msds.path
   datasetSets = msds.sets
   
   if options.smallValidSet then  -- use smaller validation set
      print('Using restricted validation set')
      datasetSets['VALID'] = msds.smallValid
   end
   if options.smallTrainSet then
      print('Using restricted train set')
      datasetSets['TRAIN'] = msds.smallTrain
   end
end

local trimPath = false  -- Changes structure of loaded filenames,
-- setting this to true uses a slightly less redundant structure, albeit more complex to use
local filenamesSets = import_dataset.import_sets_filenames(datasetPath,
							   datasetSets,
							   filter_suffix,
							   trimPath)

local auxiliary_sets = {'VALID', 'TEST'}

options['VALID'] = {}; options['TEST'] = {}

options['VALID'].subSize = #filenamesSets['VALID']
-- options['VALID'].subSize = 1000  -- Comment this out to use full validation set 

options['TEST'].subSize = #filenamesSets['TEST']
-- options['TEST'].subSize = 1000  -- Comment this out to use full validation set 

local function subrange(elems, start_idx, end_idx)
   local sub_elems = {}
   for i=start_idx, end_idx do
      table.insert(sub_elems, elems[i])
   end
   return sub_elems
end

local validFilenames, testFilenames
if not(trimPath) then
   validFilenames = subrange(filenamesSets['VALID'], 1, options['VALID'].subSize)
   testFilenames = subrange(filenamesSets['TEST'], 1, options['TEST'].subSize)
end

-- local auxiliaryData = {}
-- for _, setType in pairs(auxiliary_sets) do
--    local filenames = filenamesSets[setType]
--    print(filenames)
--    -- print(#filenames)

--    -- if options[setType].subSize < #filenames then
--    --    print('WARNING! Not using full ' .. setType:lower() ..' set!')
--    --    print('\tUsing only the first ' .. options[setType].subSize .. ' elements')
--    -- end
--    -- -- local filenames_sub = subrange(filenames, 1, options[setType].subSize)

--    -- print(filenames_sub)
--    -- Validation set as a tensor
--    if trimPath then
--       auxiliaryData[setType] = import_dataset.load_slice_filenamesFolders_tensor(
-- 	 filenames, f_load, options)
--    else
--       auxiliaryData[setType] = import_dataset.load_slice_filenames_tensor(
-- 	 filenames, f_load, options)
--    end
-- end

-- local validData = auxiliaryData['VALID']
-- local testData = auxiliaryData['TEST']

-- auxiliaryData = nil; collectgarbage(); collectgarbage()

--------------------
-- Training set
--------------------
local filenamesTrain = filenamesSets['TRAIN'];

options.numTrainFiles = #filenamesTrain
filenamesTrain = subrange(filenamesTrain, 1, options.numTrainFiles)

table.sort(filenamesTrain)

-- Randomize training examples order
local function shuffleTable(tableIn)
   return sampleFile.get_random_subset(tableIn, #tableIn)
end

-- filenamesTrain = shuffleTable(filenamesTrain)

local numFilenamesTrain = #filenamesTrain

local setList = {'MSDS'}

----------------------------------------------------------------------
-- Iterating over all potential models
----------------------------------------------------------------------

-- Unique identifier to save trained models for future use
-- (format: year_month_day-hour_minute_second)
local session_date = os.date('%Y%m%d-%H%M%S', os.time())
local savePrefix = saveFolder .. session_date .. '-'
local fd_structures = assert(io.open(savePrefix .. 'structures.txt', 'w'))

fd_structures:write('Options:\n')
for optionName, optionValue in pairs(options) do
   fd_structures:write('\t' .. optionName .. ': ' .. tostring(optionValue) .. '\n')
end
fd_structures:write('\n\n')
fd_structures:flush()

models = {modelLSTM};

-- Threshold for F0 measure:
local accCriterion_threshold = 0.2
criterions = {nn.MSECriterion,
	      nn.DistKLDivCriterion}
-- function () return nn.binaryAccCriterion(accCriterion_threshold) end

-- Sampler to use
local hyperSample = hyperSampler()
-- Parameters handling
local hyperParams = hyperParameters(hyperSample)

----------------------------------------------------------------------
-- Experiment definition code
----------------------------------------------------------------------
-- Number of repetition for each architecture
local nbRepeat = 2
-- Number of iterations
local nbSteps = 50
local nbTrainEpochs = options.datasetMaxEpochs
-- Number of networks per step
--local nbBatch = nbThreads
local nbBatch = 10
-- Full number of networks
local nbNetworks = nbSteps * nbBatch
local nbRandom = 100000

if options.manualMode then
   -- Number of repetition for each architecture
   nbRepeat = 1
   -- Number of iterations
   nbSteps = 1
   nbTrainEpochs = options.datasetMaxEpochs
   -- Number of networks per step
   --local nbBatch = nbThreads
   nbBatch = 1
   -- Full number of networks
   nbNetworks = 1
   nbRandom = 1

   models = {modelLSTM}
   criterions = {nn.MSECriterion}
end

local testing = false
if testing then
   print('WARNING, quick training enabled')
   nbSteps = 4
   nbBatch = 10
   options.maxNbLayers = 3
end

model = nil  -- Declare here for scope-compatibility (allows saving models when using manualMode)
-- Iterate over all models that we want to test
print('Start loop on models')
for k, v in ipairs(models) do   
   -- Current model to check
   local curModel = v()
   local shortModelName = 'model_' .. k .. '-' .. tostring(curModel)

   print('Current model: ' .. shortModelName)
   
   -- Define baseline parameters
   curModel:parametersDefault()
   
   print('Start loop on criterions')
   for criterion_k, criterion_v in ipairs(criterions) do
      local useLogProbabilities = false
      if criterion_v == nn.DistKLDivCriterion then
	 useLogProbabilities = true
      end
      
      local criterion = criterion_v()
      local shortCriterionName = tostring(criterion):gsub('%.', '_')
      
      print('Current criterion: ' .. shortCriterionName)
      criterion = nn.SequencerCriterion(criterion)
      criterion.name = shortCriterionName
      if options.cuda then criterion:cuda() end

      mainSaveLocation = savePrefix .. shortModelName .. '-' ..
	 shortCriterionName

      local minTestErrorNbLayers = torch.DoubleTensor(options.maxNbLayers):fill(math.huge)
      
      -- Loop over number of layers
      print('Start loop on layers number')
      for nbLayers = options.minNbLayers,options.maxNbLayers do
	 print('Current layers number: ' .. nbLayers)
	 ----------------------------------------------------------------------
	 -- Structure optimization code
	 ----------------------------------------------------------------------
	 -- Same basic structure for all
	 local structure = {}
	 structure.nLayers = nbLayers
	 structure.nInputs = options.sliceSize
	 structure.nFeats = options.featsNum
	 structure.nOutputs = options.predictionLength  -- Output only predictions	 
	 structure.maxLayerSize = 2048

	 -- Manual initialization
	 if options.manualMode then
	    -- Input chosen parameters here for manual training
	    structure.nLayers = 1
	    structure.layers = {1024, 1024, 512, 256};
	    structure.convSize = {16, 32, 64};
	    structure.kernelWidth = {8, 8, 8};
	    structure.poolSize = {2, 2, 2};
	    structure.nClassLayers = 3;
	 end
	 
	 -- Reinitialize hyperparameter optimization
	 hyperParams:unregisterAll();
	 -- Add the structure as requiring optimization
	 if options.fastRun then
	    print('WARNING: using very small layers')
	    structure.minLayerSize = 32
	    structure.maxLayerSize = 128
	 end

	 if not options.manualMode then
	    curModel:registerStructure(hyperParams, nbLayers,
				       structure.minLayerSize, structure.maxLayerSize);
	    -- Register model-specific options (e.g. non-linarity used)
	    curModel:registerOptions(hyperParams, options.fastRun)
	    -- Initialize hyperparameters structure
	    hyperParams:initStructure(nbNetworks, #setList, nbRepeat, nbSteps, nbBatch);
	 end

	 -- Optimization step
	 print('Start hyperparameters optimization loop')
	 for step = 1,nbSteps do
	    print('Current optimization step number: ' .. step .. ' of ' .. nbSteps)
	    -- Local pasts values
	    localErrors = torch.zeros(nbBatch, #setList, nbRepeat);
	    -- Architecture batch
	    for batch = 1,nbBatch do
	       -- Create a new draw of all parameters
	       hyperParams:randomDraw();
	       -- Errors of current model
	       errorRates = torch.zeros(#setList, nbRepeat);
	       -- Extract current structure
	       if not options.manualMode then
		  structure = curModel:extractStructure(hyperParams, structure);
	       end
	       -- Perform N random repetitions
	       for r = 1,nbRepeat do
		  print('Current model repetition number: ' .. r .. ' of ' .. nbRepeat)
		  for set = 1, #setList do
		     print('Current training dataset number: ' .. set .. ' of ' .. #setList)
		     -- Configure the default optimizer
		     configureOptimizer(options, resampleVal);
		     
		     -- Update model's internal parameters
		     if not options.manualMode then
			curModel:updateOptions(hyperParams)
		     end
		     -- Define a new model
		     local model = nil  -- clean any previously stored model from memory
		     collectgarbage(); collectgarbage()
		     model = curModel:defineModel(structure, options);

		     local fullManual = false
		     if options.manualMode and fullManual then
			model = nil
			if options.modelUniform then
			   -- Uniform sampling
			   require './moduleUniform.lua'
			   local modelUniform = nn.Uniform(structure.nOutputs, structure.nFeats, options)
			   model = modelUniform
			   shortModelName = 'modelUniform'
			elseif options.modelRepeat then
			   -- Memory-less repeat module
			   require './moduleRepeat.lua'
			   local modelRepeat = nn.Repeat(structure.nOutputs, options)
			   model = modelRepeat
			   shortModelName = 'modelRepeat'
			elseif options.loadModel~='' then
			   model = torch.load(options.loadModel)
			   shortModelName = 'reloaded_model'
			else
			   model = nn.Sequential()
			   
			   model:add(nn.Transpose({1, 2}))
			   model:add(nn.View(-1, structure.nInputs * structure.nFeats))

			   model:add(nn.Linear(structure.nInputs * structure.nFeats, 1024))
			   model:add(nn.BatchNormalization(1024))
			   model:add(nn.ReLU())
			   model:add(nn.Linear(1024, 1024))
			   model:add(nn.BatchNormalization(1024))
			   model:add(nn.ReLU())
			   model:add(nn.Linear(1024, 512))
			   model:add(nn.BatchNormalization(512))
			   model:add(nn.ReLU())
			   model:add(nn.Linear(512, structure.nOutputs * structure.nFeats))

			   model:add(nn.View(-1, structure.nOutputs, structure.nFeats))
			   model:add(nn.Transpose({1, 2}))
			   
			   -- TODO Should add Initializer
			   
			   shortModelName = 'manually_designed-model-mlp'
			end
			mainSaveLocation = savePrefix .. shortModelName .. '-' ..
			   shortCriterionName
		     end
		     
		     -- Add a LogSoftMax layer to compute log-probabilities for KL-Divergence
		     if useLogProbabilities then
			model:add(nn.Sequencer(nn.LogSoftMax()))
		     end
		     
		     -- Activate CUDA on the model
		     if options.cuda then model:cuda(); end
		     
		     if not(options.manualMode and fullManual or options.loadModel) then
			model = curModel:weightsInitialize(model)
		     end
		     print('Current model:')
		     print(model)
		     if not options.manualMode then hyperParams:printCurrent() end
		     ----------------------------------------------------------------------
		     -- Prediction training code
		     ----------------------------------------------------------------------

		     -- Flatten all trainable parameters into a 1-dim vector
		     if model then
			parameters, gradParameters = curModel:getParameters(model);
		     end
		     
		     local fullModelString = tostring(model):gsub('%.', '_')
		     if r == 1 then  -- Don't rewrite structure at each training repetition
			fd_structures:write('Network structure ' .. shortModelName ..
					       ':\n' .. fullModelString .. '\n\n')
			fd_structures:flush()
		     end
		     
		     local validIncreasedEpochs, initialValidError, minValidErr 
		     if options.printValidation then
			-- Keep track of best validation error
			local validIncreasedEpochs = 0
			print('Computing initial validation error:')
			local initialValidError = slidingTest(model, criterion, validFilenames,
							      f_load, options, hyperParams);
			print('\nInitial valid error: ' .. initialValidError)
			local minValidErr = initialValidError
		     end
		     
		     local validationPrinter = getValidationPrinter(model, criterion,
								    validFilenames, f_load,
								    options, hyperParams)
		     -----------------------------------------------------------
		     -- DISABLED: No loop on layers, uses batch-normalize --
		     -----------------------------------------------------------
		     --for l = 1, structure.nLayers do
		     -- print('Start loop on layers')
		     -- To save intermediate results during learning
		     -- local layerSaveLocation = mainSaveLocation ..
		     --    '-pretrain-layer_' .. l .. '.net'
		     -- Define the pre-training model
		     -- DISABLED: use batch-normalize instead
		     -- local model = curModel:definePretraining(structure, l, options);

		     local saveLocation = layerSaveLocation or mainSaveLocation .. '.net'
		     -- If classical learning configure the optimizer
		     -- TODO
		     -- if (not options.adaptiveLearning) then 
		     -- 	 if torch.type(unsupData.data) ~= 'table' then
		     -- 	    configureOptimizer(options, unsupData.data:size(2))
		     -- 	 else
		     -- 	    configureOptimizer(options, #unsupData.data);
		     -- 	 end
		     -- end
		     
		     local isNaNError = false
		     -- Perform SGD on this subset of the dataset
		     print('Start loop on dataset windows')
		     for datasetEpoch = 1, nbTrainEpochs do
			if options.testOnly then break end
			if isNaNError then break end
			print('Current training epoch: ' .. datasetEpoch .. ' of ' .. nbTrainEpochs)
			-- Perform sliding window over the training dataset (too large to fit in memory)
			for slice, cur_start, cur_end in import_dataset.get_sliding_window_iterator(
			   filenamesTrain, options.datasetWindowSize, options.datasetWindowStep,
			   f_load, options) do
			   xlua.progress(cur_start, numFilenamesTrain)
			   
			   validationPrinter(cur_start)
			   
			   local trainData = {}
			   -- Slice examples into small training examples with size depending on the
			   -- hyper-parameters
			   -- for dataType, dataSubset in pairs(slice) do
			   --    -- Iterate over inputs and targets
			   --    local smallSlidingWindowBatch = batchSlidingWindow(dataSubset)
			   --    trainData[dataType] = smallSlidingWindowBatch
			   -- end
			   trainData = slice
			   
			   if (not options.adaptiveLearning) then
			      if torch.type(trainData.data) ~= 'table' then
				 configureOptimizer(options, trainData.data:size(options.batchDim))
			      else
				 configureOptimizer(options, #trainData.data);
			      end
			   end

			   --[[ Adaptive learning ]]--
			   if options.adaptiveLearning then
			      -- 1st epochs = Start with purely stochastic (SGD) on single examples
			      if epoch == 0 then
				 configureOptimizer({optimization = 'SGD', batchSize = 5,
						     learningRate = 5e-3},
				    trainData.data:size(2));
			      end
			      -- Next epochs = Sub-linear approximate algorithm ASGD with mini-batches
			      if epoch == options.subLinearEpoch then
				 configureOptimizer({optimization = 'SGD', batchSize = 128,
						     learningRate = 2e-3},
				    trainData.data:size(2))
			      end
			      -- Remaining epochs = Advanced learning algorithm user-selected
			      -- (LBFGS | CG | ADADELTA | ADAGRAD | ADAM | ADAMAX |
			      --  FISTALS | NAG | RMSPROP | RPROP | CMAES)
			      if epoch == options.superLinearEpoch then
				 configureOptimizer(options, trainData.data:size(options.batchDim))
			      end
			   end
			   
			   err = unsupervisedTrain(model, criterion, trainData, options);
			   print("Reconstruction error (train): " .. err);

			   if _.isNaN(err) then
			      isNaNError= true 
			      print("Training error is NaN: divergent error, stop training");
			      break
			   end
			   
			   -- Collect the garbage
			   collectgarbage(); collectgarbage();
			end

			-- Disabled early-stopping, models are too big to reload...
			
			-- 	--[[ Validation set checking/Early-stopping ]]--
			-- 	if datasetEpoch % options.validationRate == 0 then
			-- 	   -- Check reconstruction error on the validation data
			-- 	   print('Computing validation error:')
			-- 	   validErr = slidingTest(model, criterion, validFilenames,
			-- 				  f_load, options, hyperParams);
			-- 	   print("Reconstruction error (valid) : " .. validErr);

			-- 	   if validErr > minValidErr then
			-- 	      -- The validation error has risen since last checkpoint

			-- 	      validIncreasedEpochs = validIncreasedEpochs + 1
			-- 	      print('Validation error increased, previous best value was ' ..
			-- 		       validIncreasedEpochs .. ' epochs ago')

			-- 	      if validIncreasedEpochs > options.maxValidIncreasedEpochs then
			-- 		 -- Reload the last saved model
			-- 		 model = torch.load(saveLocation);
			-- 		 -- Stop the learning
			-- 		 print(" => Stop learning");
			-- 		 break;
			-- 	      end
			-- 	   else
			-- 	      -- Keep the current error as reference
			-- 	      minValidErr = validErr;
			-- 	      -- Save the current best model
			-- 	      torch.save(saveLocation, model);

			-- 	      validIncreasedEpochs = 0
			-- 	   end
			-- 	end
			if options.manualMode then
			   torch.save(mainSaveLocation .. "-trained_model.dat", model)
			end
		     end

		     -- if not testing then
		     -- 	-- Retrieve best training iteration for the current model
		     -- 	model = torch.load(saveLocation);
		     -- 	os.remove(saveLocation)  -- clean saved networks
		     -- end
		     
		     -- Evaluate test error
		     
		     -- set = number of datasets => 1
		     -- r = number of repetitions (because random draws, keep best)
		     -- r could be set to one for a quick run
		     -- errorRates is then a 1x1 Tensor
		     print('Computing test error:')
		     
		     local testError
		     -- Make sure never to store a NaN test error, store an inf error instead
		     -- for consistency
		     if not isNaNError then  -- Avoid computation if model has diverged
			testError = slidingTest(model, criterion, testFilenames,
						f_load, options, hyperParams)
			print('\nReconstruction error (test): ' .. testError)
			print('\tMean reconstruction error (test): ' .. testError / #testFilenames)
		     end
		     if not(testError) or _.isNaN(testError) then
			print('\tTest error was NaN, setting it to +inf')
			testError = math.huge
		     end
		     
		     errorRates[{set, r}] = testError
		     
		     -- Update layer-wise minimal error
		     minTestErrorNbLayers[nbLayers] = math.min(
			minTestErrorNbLayers[nbLayers], testError)
		     
		     -- Remove garbage
		     collectgarbage(); collectgarbage();
		  end -- set
	       end  -- repeat
	       hyperParams:registerResults(errorRates);
	    end  -- batch

	    if not options.manualMode then
	       -- Rank different architecture against each other
	       hyperParams:updateRanks();
	       -- Find the next values of parameters to evaluate
	       hyperParams:fit(nbRandom, nbBatch);
	       -- Save the current state of optimization (only errors and structures)
	       resultsFilename_extensionFree = mainSaveLocation .. "-optimize_" .. nbLayers
	       fID = assert(io.open(resultsFilename_extensionFree .. '.txt', "w"));
	       hyperParams:outputResults(fID);
	       fID:close();
	       hyperParams:saveResults(resultsFilename_extensionFree .. '.dat');
	    end
	 end  -- step
	 if options.manualMode then break end
      end  -- nbLayers
      if not options.manualMode then
	 torch.save(mainSaveLocation .. "-min_test_error_per_nb_layers.dat", minTestErrorNbLayers)
      end
   end  -- criterion
end  -- model

fd_structures:close()
