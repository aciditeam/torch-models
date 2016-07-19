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

-- TODO
-- cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
-- cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
-- cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
-- cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
-- TODO

-- parse input params
cmd_params = cmd:parse(arg)

local options = ts_init.get_options(cmd_params.useCuda)

local function append(appTable, mainTable)
   for k, v in pairs(appTable) do
      mainTable[k] = appTable[k]
   end
end
append(cmd_params, options)

-- RNN-library's batch conventions
options.tDim = 1
options.batchDim = 2
options.featsDim = 3

ts_init.set_globals(); ts_init.set_cuda(options)

-- All sequences will be sliced into sub-sequences of this duration
options.sliceSize = 128

-- Not all the dataset is loaded into memory in a single pass,
-- we perform a sliding window over it, load a subset, perform
-- some iterations of the optimisation process, then slide the window
options.datasetWindowSize = 128
options.datasetMaxEpochs = 30
options.datasetWindowStepSize = math.floor(options.datasetWindowSize / 2) 

-- Maximum number of successive times the validation error can increase
-- before stopping training
options.maxValidIncreasedEpochs = 5

-- Training parameters
options.batchSize = 64;

---------------------------------------
-- Length of predictions to perform
---------------------------------------
options.predict = true
options.predictionLength = 1

if options.predictionLength ~= 1 then
   error("Support for prediction length ~= 1 not yet implemented." ..
	    'Best would be to slice target predictions within the networks.')
end

-- Initialize a sliding window depending on hyperparameters
local function getSlidingWindow(hyperParams)
   local function batchSlidingWindow(x) return x end

   local parameters = moduleSequencerSlidingWindow.getParameters(hyperParams)
   -- Smaller-sized sliding window over a batch of long examples
   local slidingWindow = nn.SequencerSlidingWindow(
      1, parameters.slidingWindowSize, parameters.slidingWindowStep)
   batchSlidingWindow = function(minibatch)
      return slidingWindow:forward(minibatch)
   end
   return batchSlidingWindow
end

-- Extract only actual predictions from a batch of examples
-- Sequence loader returns targets as simply offset versions of their data
-- counterpart, this extracts the predictionLength tail of those sequences
local function selectPrediction(batch, options)
   local sliceSize = batch:size(options.tDim)
   return batch:narrow(1, sliceSize-options.predictionLength+1, options.predictionLength)
end

-- Debug and printing parameters
-- Print current validation every ... analyzed files
options.printValidationRate = 200
-- Return a function periodically printing validation error of model of data
local function getValidationPrinter(model, validData, options)
   local previous_file_position = 1
   local previous_print_valid = 0
   
   local function validationPrinter(file_position)
      local loaded_files = file_position - previous_file_position
      previous_print_valid = previous_print_valid + loaded_files
      
      if previous_print_valid >= options.printValidationRate then
	 validErr = unsupervisedTest(model, validData,
				     options);
	 print('Current validation error before training: ' .. validErr)
	 previous_print_valid = previous_print_valid %
	    options.printValidationRate
      end
      previous_file_position = file_position
   end
   return validationPrinter
end

----------------------------------------------------------------------
-- Initialize datasets
----------------------------------------------------------------------

local msds = require './importMSDS'

local f_load = msds.load.get_btchromas

-- local _, filenamesSets = ts_init.import_data(baseDir, setList, options)
local filter_suffix = '.dat'
local filenamesSets = import_dataset.import_sets_filenames(msds.subset.path,
							   msds.subset.sets,
							   filter_suffix)

local auxiliary_sets = {'VALID', 'TEST'}

options['VALID'] = {}; options['TEST'] = {}   

options['VALID'].subSize = #filenamesSets['VALID']
--options['VALID'].subSize = 300  -- Comment this out to use full validation set 

options['TEST'].subSize = #filenamesSets['TEST']
--options['TEST'].subSize = 300  -- Comment this out to use full validation set 

local function subrange(elems, start_idx, end_idx)
   local sub_elems = {}
   for i=start_idx, end_idx do
      table.insert(sub_elems, elems[i])
   end
   return sub_elems
end

local auxiliaryData = {}
for _, setType in pairs(auxiliary_sets) do
   local filenames = filenamesSets[setType]
   print(#filenames)

   if options[setType].subSize < #filenames then
      print('WARNING! Not using full ' .. setType:lower() ..' set!')
      print('\tUsing only the first ' .. options[setType].subSize .. ' elements')
   end
   local filenames_sub = subrange(filenames, 1, options[setType].subSize)
   
   -- Validation set as a tensor
   auxiliaryData[setType] = import_dataset.load_slice_filenames_tensor(
      filenames_sub, f_load, options)
end

local validData = auxiliaryData['VALID']
local testData = auxiliaryData['TEST']

options.featsNum = validData.data:size(options.featsDim)

auxiliaryData = nil; collectgarbage(); collectgarbage()

--------------------
-- Training set
--------------------
local filenamesTrain = filenamesSets['TRAIN'];

-- Randomize training examples order
local function shuffleTable(tableIn)
   return sampleFile.get_random_subset(tableIn, #tableIn)
end

filenamesTrain = shuffleTable(filenamesTrain)

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
criterions = {nn.DistKLDivCriterion,
	      nn.MSECriterion,
	      function () return nn.binaryAccCriterion(accCriterion_threshold) end}

print('WARNING must be careful to correcty store results for different criterions')

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
local nbSteps = 100
local nbTrainSteps = 5
local nbTrainNetworks = 10
local nbTrainEpochs = options.datasetMaxEpochs
-- Number of networks per step
--local nbBatch = nbThreads
local nbBatch = 10
-- Full number of networks
local nbNetworks = nbSteps * nbBatch
local nbRandom = 100000;


local testing = false
if testing then
   print('WOOOOOOOOOOOOOW, mega hard training dude')
   nbSteps = 1
   nbBatch = 1
end

-- Iterate over all models that we want to test
for k, v in ipairs(models) do   
   print('Start loop on models')
   -- Current model to check
   curModel = v()
   local shortModelName = 'model_' .. k .. '-' .. tostring(curModel)

   print('Current model: ' .. shortModelName)
   
   -- Define baseline parameters
   curModel:parametersDefault()
   
   for criterion_k, criterion_v in ipairs(criterions) do
      print('Start loop on criterions')
      criterion = criterion_v()
      local shortCriterionName = 'criterion_' .. k .. '-' ..
	 tostring(criterion):gsub('%.', '_')
      
      print('Current criterion: ' .. shortCriterionName)
      criterion = nn.SequencerCriterion(criterion)
      if options.cuda then criterion:cuda() end

      local mainSaveLocation = savePrefix .. '-' .. shortModelName .. '-' ..
	 shortCriterionName
      
      -- Loop over number of layers
      for nbLayers = 3,8 do
	 ----------------------------------------------------------------------
	 -- Structure optimization code
	 ----------------------------------------------------------------------
	 -- Same basic structure for all
	 local structure = {}
	 structure.nLayers = nbLayers
	 structure.nInputs = options.featsNum
	 structure.nOutputs = structure.nInputs;	 

	 -- -- Default initialization
	 -- structure.nLayers = 3
	 -- structure.nInputs = 12
	 -- structure.layers = {100, 100, 50, 20};
	 -- structure.nOutputs = structure.nInputs;
	 -- structure.convSize = {16, 32, 64};
	 -- structure.kernelWidth = {8, 8, 8};
	 -- structure.poolSize = {2, 2, 2};
	 -- structure.nClassLayers = 3;
	 
	 -- Reinitialize hyperparameter optimization
	 hyperParams:unregisterAll();
	 -- Add the structure as requiring optimization
	 local minSize, maxSize
	 if options.fastRun then
	    print('WAAAAAAAARNING, USING SUPER SMALL LAYERS')
	    minSize = 32
	    maxSize = 128
	    print('WAAAAAAAARNING, USING SUPER SMALL MEMORY')
	 end


	 -- Register external parameters for sliding window
	 moduleSequencerSlidingWindow.registerParameters(hyperParams)

	 curModel:registerStructure(hyperParams, nbLayers, minSize, maxSize);
	 -- Register model-specific options (e.g. non-linarity used)
	 curModel:registerOptions(hyperParams, options.fastRun)
	 -- Initialize hyperparameters structure
	 hyperParams:initStructure(nbNetworks, #setList, nbRepeat, nbSteps, nbBatch);
	 
	 -- Optimization step
	 for step = 1,nbSteps do
	    -- Local pasts values
	    localErrors = torch.zeros(nbBatch, #setList, nbRepeat);
	    -- Architecture batch
	    for batch = 1,nbBatch do
	       -- Create a new draw of all parameters
	       hyperParams:randomDraw();
	       -- Errors of current model
	       errorRates = torch.zeros(#setList, nbRepeat);
	       -- Extract current structure
	       structure = curModel:extractStructure(hyperParams, structure);
	       -- Perform N random repetitions
	       for r = 1,nbRepeat do
		  for set = 1, #setList do 
		     -- Configure the default optimizer
		     configureOptimizer(options, resampleVal);
		     
		     -- Update model's internal parameters
		     curModel:updateOptions(hyperParams)
		     -- Define a new model
		     model = curModel:defineModel(structure, options);
		     		     
		     -- Activate CUDA on the model
		     if options.cuda then model:cuda(); end

		     model = curModel:weightsInitialize(model)
		     print(model)
		     hyperParams:printCurrent()
		     ----------------------------------------------------------------------
		     -- Prediction training code
		     ----------------------------------------------------------------------

		     -- Flatten all trainable parameters into a 1-dim vector
		     if model then
			parameters, gradParameters = curModel:getParameters(model);
		     end
		     
		     local fullModelString = tostring(model):gsub('%.', '_')
		     fd_structures:write('Network structure ' .. shortModelName ..
					    ':\n' .. fullModelString .. '\n\n')
		     fd_structures:flush()
		     
		     -- Set of trained layers
		     trainedLayers = {};
		     
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


		     local batchSlidingWindow = getSlidingWindow(hyperParams)
		     
		     -- Keep track of best validation error
		     local validIncreasedEpochs = 0
		     local initialValidError = unsupervisedTest(model, validData,
								options);
		     print('Initial valid error: ' .. initialValidError)
		     local minValidErr = initialValidError

		     -- Perform SGD on this subset of the dataset
		     for datasetEpoch = 0, nbTrainEpochs do

			local validationPrinter = getValidationPrinter(model, validData,
								       options)
			
			if testing then
			   print('WOOOOOW, hardcore testing m8')
			   break
			end

			print('Start loop on dataset windows')
			-- Perform sliding window over the training dataset (too large to fit in memory)
			for slice, file_position in import_dataset.get_sliding_window_iterator(
			   filenamesTrain, f_load, options) do
			   xlua.progress(file_position, numFilenamesTrain)
			   
			   validationPrinter(file_position)
			   
			   local trainData = {}
			   -- Slice examples them into small training examples with size
			   -- options.slidingWindowSize
			   for dataType, dataSubset in pairs(slice) do
			      -- Iterate over inputs and targets
			      local smallSlidingWindowBatch = batchSlidingWindow(dataSubset)
			      trainData[dataType] = smallSlidingWindowBatch
			   end
			   
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
			   
			   err = unsupervisedTrain(model, trainData, options);
			   print("Reconstruction error (train) : " .. err);
			   
			   -- Collect the garbage
			   collectgarbage();
			end

			--[[ Validation set checking/Early-stopping ]]--
			if datasetEpoch % options.validationRate == 0 then
			   -- Check reconstruction error on the validation data
			   validErr = unsupervisedTest(model, validData,
						       options);
			   print("Reconstruction error (valid) : " .. validErr);

			   if validErr > minValidErr then
			      -- The validation error has risen since last checkpoint

			      validIncreasedEpochs = validIncreasedEpochs + 1
			      print('Validation error increased, previous best value was ' ..
				       validIncreasedEpochs .. ' epochs ago')

			      if validIncreasedEpochs > options.maxValidIncreasedEpochs then
				 -- Reload the last saved model
				 model = torch.load(saveLocation);
				 -- Stop the learning
				 print(" => Stop learning");
				 break;
			      end
			   else
			      -- Keep the current error as reference
			      minValidErr = validErr;
			      -- Save the current best model
			      torch.save(saveLocation, model);

			      validIncreasedEpochs = 0
			   end
			end
		     end

		     if not testing then
			-- Retrieve best training iteration for the current model
			model = torch.load(saveLocation);
			os.remove(saveLocation)  -- clean saved networks
		     end

		     -- Evaluate test error
		     
		     -- set = number of datasets => 1
		     -- r = number of repetitions (because random draws, keep best)
		     -- r could be set to one for a quick run
		     -- errorRates is then a 1x1 Tensor
		     errorRates[{set, r}] = unsupervisedTest(model, testData, options)
		     
		     -- Remove garbage
		     collectgarbage();
		  end -- set
	       end  -- repeat
	       hyperParams:registerResults(errorRates);
	    end  -- batch
	    
	    -- Rank different architecture against each other
	    hyperParams:updateRanks();
	    -- Find the next values of parameters to evaluate
	    hyperParams:fit(nbRandom, nbBatch);
	    -- Save the current state of optimization (only errors and structures)
	    fID = assert(io.open(mainSaveLocation .. "-optimize_" .. nbLayers .. ".txt", "w"));
	    hyperParams:outputResults(fID);
	    fID:close();
	 end  -- step
      end  -- nbLayers
   end  -- criterion
end  -- modelClass

fd_structures:close()
