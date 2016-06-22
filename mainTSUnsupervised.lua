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
require 'SequencerSlidingWindow'
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

----------------------------------------------------------------------
-- Initialization
--
-- Need to initialiaze: option, unSets
----------------------------------------------------------------------

local ts_init = require './TSInitialize'

local saveFolder = '/data/Documents/machine_learning/models/time_series/'

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
options.datasetWindowSize = 64
options.datasetMaxEpochs = 100
options.datasetWindowStepSize = math.floor(options.datasetWindowSize / 2) 

-- Maximum number of successive times the validation error can increase
-- before skipping to the next subset of training data
options.maxValidIncreasedEpochs = 5

-- Training parameters
options.batchSize = 32;

-- Use a sliding window on the sequences to train on a lot of small sequences
options.slidingWindowSize = 32;
options.slidingWindowStep = 16;

-- Train to predict next steps
options.predict = true


-- Smaller-sized sliding window over a batch of long examples
local slidingWindow = nn.SequencerSlidingWindow(1, options.slidingWindowSize,
						options.slidingWindowStep)
local function batchSlidingWindow(minibatch)
   return slidingWindow:forward(minibatch)
end

----------------------------------------------------------------------
-- Initialize datasets
----------------------------------------------------------------------

local msds = require './importMSDS'

-- local _, filenamesSets = ts_init.import_data(baseDir, setList, options)
local filter_suffix = '.h5'
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

options.validSubSize = 100
local filenamesValid_sub = subrange(filenamesValid, 1, options.validSubSize)

local f_load = msds.load.get_btchromas
-- Validation set as a tensor
print(filenamesValid_sub)
local unsupValid_data, unsupValid_targets = import_dataset.load_slice_filenames_tensor(
   filenamesValid_sub, f_load, options)
local unsupValid = {data = unsupValid_data,
		    targets = unsupValid_targets}

print(unsupValid_data:size())

----------------------------------------------------------------------
-- Iterating over all potential models
----------------------------------------------------------------------

-- Unique identifier to save trained models for future use
-- (format: year_month_day-hour_minute_second)
local session_date = os.date('%Y%m%d-%H%M%S', os.time())
local savePrefix = saveFolder .. 'results/' .. session_date .. '-'
local fd_structures = assert(io.open(savePrefix .. 'structures.txt', 'w'))

fd_structures:write('Options:\n')
for optionName, optionValue in pairs(options) do
   fd_structures:write('\t' .. optionName .. ': ' .. tostring(optionValue) .. '\n')
end
fd_structures:write('\n\n')
fd_structures:flush()

-- models = {modelMLP, modelCNN, modelInception, modelVGG};
models = {modelLSTM};
-- TODO: add Boulanger-Lewandowsky's distance
criterions = {nn.MSECriterion, nn.DistKLDivCriterion}

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
   structure.nLayers = 1;  -- TODO: add more layers
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
   -- Unsupervised training code
   ----------------------------------------------------------------------
   -- TODO
   -- TODO
   -- Sub-loop on hyper-parameter optimization !
   -- Even for pre-training
   -- TODO
   -- TODO
   
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
      
      -- -- TODO: Temporary model redefinition
      -- model, criterion = nil
      -- model = nn.SeqLSTM(12, 12)
      -- criterion = nn.MSECriterion()
      -- criterion = nn.SequencerCriterion(criterion)
      
      local fullModelString = tostring(model):gsub('%.', '_')
      fd_structures:write('Network structure ' .. shortModelName ..
			     ':\n' .. fullModelString .. '\n\n')
      fd_structures:flush()
      
      -- Set of trained layers
      trainedLayers = {};
      
      -- Unsupervised set
      local filenamesTrain = filenamesSets["TRAIN"];
      
      -- -- Switch training data to GPU
      -- if options.cuda then
      --    unsupData.data = unsupData.data:cuda();
      --    unsupValid.data = unsupValid.data:cuda();
      -- end
      
      for l = 1, structure.nLayers do
	 print('Start loop on layers')
	 -- To save intermediate results during learning
	 local saveLocation = savePrefix .. shortModelName ..
	    shortCriterionName ..'-pretrain-layer_' .. l .. '.net'
	 
	 -- Define the pre-training model
	 -- TODO: disabled this for compatibility, re-enable eventually
	 -- local model = curModel:definePretraining(structure, l, options);

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

	 -- Keep track of best validation error
	 local minValidErr = math.huge
	 
	 local datasetEpoch
	 -- Perform SGD on this subset of the dataset
	 for datasetEpoch = 1, options.datasetMaxEpochs do
	    local f_load = msds.load.get_btchromas

	    -- Perform sliding window over the training dataset (too large to fit in memory)
	    for slices in import_dataset.get_sliding_window_iterator(
	       {TRAIN = filenamesTrain}, f_load, options) do
	       print(collectgarbage('count'))
	       -- print('Start loop on dataset windows')
	       local validIncreasedEpochs = 0

	       local miniSequences = {}
	       -- Take a random subset of examples and slice them into
	       -- small training examples with size options.slidingWindowSize
	       for dataType, dataSubset in pairs(slices['TRAIN']) do
		  -- Iterate over inputs and targets
		  local smallSlidingWindowBatch = batchSlidingWindow(dataSubset)
		  miniSequences[dataType] = smallSlidingWindowBatch
	       end

	       print('Dataset window of size: ')
	       print(miniSequences['data']:size())

	       for epoch=0, options.maxEpochs do
		  print("Epoch #" .. epoch);
		  print(collectgarbage('count'))
		  
		  -- Create minibatch
		  local unsupData = {}
		  -- Gets random indexes for both inputs and targets
		  local indexes = torch.randperm(slices['TRAIN']['data']:size(options.batchDim)):
		     sub(1, options.batchSize):long()

		  for dataType, dataSubset in pairs(miniSequences) do
		     unsupData[dataType] = miniSequences[dataType]:index(options.batchDim, indexes)
		  end
		  
		  -- -- Take a random subset of examples and slice them into
		  -- -- small training examples with size options.slidingWindowSize
		  -- for dataType, dataSubset in pairs(slices['TRAIN']) do
		  -- 	  -- Iterate over inputs and targets
		  -- 	  local shuffledExamplesSubset = dataSubset:index(options.batchDim, indexes)
		  -- 	  local smallSlidingWindowBatch = batchSlidingWindow(
		  -- 	     shuffledExamplesSubset)
		  -- 	  unsupData[dataType] = smallSlidingWindowBatch
		  -- end
		  
		  if (not options.adaptiveLearning) then
		     if torch.type(unsupData.data) ~= 'table' then
			configureOptimizer(options, unsupData.data:size(options.batchDim))
		     else
			configureOptimizer(options, #unsupData.data);
		     end
		  end

		  --[[ Adaptive learning ]]--
		  if options.adaptiveLearning then
		     -- 1st epochs = Start with purely stochastic (SGD) on single examples
		     if epoch == 0 then
			configureOptimizer({optimization = 'SGD', batchSize = 5,
					    learningRate = 5e-3},
			   unsupData.data:size(2));
		     end
		     -- Next epochs = Sub-linear approximate algorithm ASGD with mini-batches
		     if epoch == options.subLinearEpoch then
			configureOptimizer({optimization = 'SGD', batchSize = 128,
					    learningRate = 2e-3},
			   unsupData.data:size(2))
		     end
		     -- Remaining epochs = Advanced learning algorithm user-selected
		     -- (LBFGS | CG | ADADELTA | ADAGRAD | ADAM | ADAMAX |
		     --  FISTALS | NAG | RMSPROP | RPROP | CMAES)
		     if epoch == options.superLinearEpoch then
			configureOptimizer(options, unsupData.data:size(options.batchDim))
		     end
		  end
		  
		  --[[ Unsupervised pre-training ]]--
		  -- Perform unsupervised training of the model
		  -- err = curModel:
		  err = unsupervisedTrain(model, unsupData, epoch, options);
		  print("Reconstruction error (train) : " .. err);
		  
		  --[[ Validation set checking ]]--
		  if epoch % options.validationRate == 0 then
		     -- Check reconstruction error on the validation data
		     validErr = unsupervisedTest(model, unsupValid,
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
		  -- Collect the garbage
		  collectgarbage();
	       end
	    end
	 end
	 
	 -- Keep trained layer in table
	 trainedLayers[l] = model;
	 -- Retrieve the encoding layer only
	 model = curModel:retrieveEncodingLayer(model)
	 -- Put model in evaluation mode
	 model:evaluate();
	 -- Prepare a set of activations
	 forwardedData = {data = {}};
	 forwardedValid = {data = {}};
	 
	 -- Perform forward propagation on data
	 forwardedData.data = model:forward(unsupData.data);
	 if torch.type(forwardedData.data) ~= 'table' then
	    forwardedData.data = forwardedData.data:clone()
	 else
	    for i = 1,#forwardedData.data do
	       forwardedData.data[i] = forwardedData.data[i]:clone()
	    end
	 end
	 
	 -- Replace previous set
	 unsupData = forwardedData;
	 -- Perform forward propagation on validation
	 forwardedValid.data = model:forward(unsupValid.data);
	 if torch.type(forwardedValid.data) ~= 'table' then
	    forwardedValid.data = forwardedValid.data:clone()
	 else
	    for i = 1,#forwardedValid.data do
	       forwardedValid.data[i] = forwardedValid.data[i]:clone()
	    end
	 end

	 -- Replace previous set
	 unsupValid = forwardedValid;
	 -- Remove garbage
	 collectgarbage();
      end
   end
end

fd_structures:close()
