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

-- TODO
-- cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
-- cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
-- cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
-- cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
-- TODO

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

-- Not all the dataset is loaded into memory in a single pass,
-- we perform a sliding window over it, load a subset, perform
-- some iterations of the optimisation process, then slide the window
options.datasetWindowSize = 128
options.datasetMaxEpochs = 100
options.datasetWindowStepSize = math.floor(options.datasetWindowSize / 2) 

-- Maximum number of successive times the validation error can increase
-- before skipping to the next subset of training data
options.maxValidIncreasedEpochs = 5

-- Training parameters
options.batchSize = 64;

-- Use a sliding window on the sequences to train on a lot of small sequences
options.slidingWindow = true
options.slidingWindowSize = 32;
options.slidingWindowStep = 16;

-- Train to predict next steps
options.predict = true

-- Initialize a sliding window depending on a set of options
local function getSlidingWindow(hyperParams)
   local function batchSlidingWindow(x) return x end

   if options.slidingWindow then
      -- Smaller-sized sliding window over a batch of long examples
      local slidingWindow = nn.SequencerSlidingWindow(1, options.slidingWindowSize,
						      options.slidingWindowStep)
      batchSlidingWindow = function(minibatch)
	 return slidingWindow:forward(minibatch)
      end
   end
   return batchSlidingWindow
end

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

-- local _, filenamesSets = ts_init.import_data(baseDir, setList, options)
local filter_suffix = '.dat'
local filenamesSets = import_dataset.import_sets_filenames(msds.subset.path,
							   msds.subset.sets,
							   filter_suffix)

local filenamesValid = filenamesSets['VALID']

local function subrange(elems, start_idx, end_idx)
   local sub_elems = {}
   for i=start_idx, end_idx do
      table.insert(sub_elems, elems[i])
   end
   return sub_elems
end

options.validSubSize = #filenamesValid
options.validSubSize = 300  -- Comment this to use full validation set 

if options.validSubSize < #filenamesValid then
   print('WARNING! Not using full validation set!')
   print('Using only the first ' .. options.validSubSize .. ' elements')
end
local filenamesValid_sub = subrange(filenamesValid, 1, options.validSubSize)

local f_load = msds.load.get_btchromas

-- Validation set as a tensor
local validData = import_dataset.load_slice_filenames_tensor(
   filenamesValid_sub, f_load, options)

-- Training set
local filenamesTrain = filenamesSets['TRAIN'];

-- Randomize training examples order
local function shuffleTable(tableIn)
   return sampleFile.get_random_subset(tableIn, #tableIn)
end

filenamesTrain = shuffleTable(filenamesTrain)

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
criterions = {nn.MSECriterion, nn.DistKLDivCriterion,
	      function () return nn.binaryAccCriterion(accCriterion_threshold) end}

-- Iterate over all models that we want to test
for k, v in ipairs(models) do   
   print('Start loop on models')
   -- Current model to check
   curModel = v();
   local shortModelName = 'model_' .. k .. '-' .. tostring(curModel)

   print('Current model: ' .. shortModelName)
   
   -- Define baseline parameters
   curModel:parametersDefault();
   ----------------------------------------------------------------------
   -- Structure optimization code
   ----------------------------------------------------------------------
   -- Define structure
   structure = {};
   -- Default initialization
   structure.nLayers = 3;  -- TODO: add more layers
   structure.nInputs = 12;
   structure.layers = {100, 100, 50, 20};
   structure.nOutputs = structure.nInputs;
   structure.convSize = {16, 32, 64};
   structure.kernelWidth = {8, 8, 8};
   structure.poolSize = {2, 2, 2};
   structure.nClassLayers = 3;
   
   -- TODO
   -- TODO
   -- Here we should start by
   --   * Random structure optimization
   --   * (Partly pre-trained ?)
   -- TODO
   -- TODO

   ----------------------------------------------------------------------
   -- Prediction training code
   ----------------------------------------------------------------------
   
   -- Define the model
   model = curModel:defineModel(structure, options);

   for criterion_k, criterion_v in ipairs(criterions) do
      print('Start loop on criterions')
      criterion = criterion_v()
      local shortCriterionName = 'criterion_' .. k .. '-' ..
	 tostring(criterion):gsub('%.', '_')

      print('Current criterion: ' .. shortCriterionName)
      
      criterion = nn.SequencerCriterion(criterion)
      
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
      
      local mainSaveLocation = savePrefix .. shortModelName ..
	 shortCriterionName
      
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
      
      -- Activate CUDA on the model
      if options.cuda then model:cuda(); end
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
      local initialValidError = unsupervisedTest(model, validData,
						 options);
      print('Initial validation error: ' .. initialValidError)
      local validIncreasedEpochs = 0
      local minValidErr = initialValidError

      local datasetEpoch
      -- Perform SGD on this subset of the dataset
      for datasetEpoch = 0, options.datasetMaxEpochs do
	 xlua.progress(datasetEpoch, options.datasetMaxEpochs)

	 local validationPrinter = getValidationPrinter(model, validData,
							options)
	 
	 print('Start loop on dataset windows')
	 -- Perform sliding window over the training dataset (too large to fit in memory)
	 for slice, file_position in import_dataset.get_sliding_window_iterator(
	    filenamesTrain, f_load, options) do
	    print('Last loaded file, number: ' .. file_position)

	    validationPrinter(file_position)
	    
	    local trainData = {}
	    -- Take a random subset of examples and slice them into
	    -- small training examples with size options.slidingWindowSize
	    for dataType, dataSubset in pairs(slice) do
	       -- Iterate over inputs and targets
	       local smallSlidingWindowBatch = batchSlidingWindow(dataSubset)
	       trainData[dataType] = smallSlidingWindowBatch
	    end
	    
	    if options.cuda then
	       for dataType, _ in pairs(slice) do
		  -- Iterate over inputs and targets
		  trainData[dataType]:cuda()
	       end
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
	    
	    --[[ Unsupervised pre-training ]]--
	    -- Perform unsupervised training of the model
	    -- err = curModel:
	    err = unsupervisedTrain(model, trainData, datasetEpoch, options);
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
      -- end

      -- Remove garbage
      collectgarbage();
   end
end
-- end

fd_structures:close()
