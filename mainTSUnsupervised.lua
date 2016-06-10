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

local saveFolder = '/data/Documents/machine_learning/models/time_series-temp/'

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

ts_init.set_globals(); ts_init.set_cuda(options)

options.batchSize = 16;

-- Train to predict next steps
options.predict = true

-- All sequences will be sliced into sub-sequences of this duration
options.sliceSize = 128

-- Not all the dataset is loaded into memory in a single pass,
-- we perform a sliding window over it, load a subset, perform
-- some iterations of the optimisation process, then slide the window
options.datasetWindowSize = 256
options.datasetMaxEpochs = 100
options.datasetWindowStepSize = math.floor(options.datasetWindowSize / 2) 

-- Maximum number of successive times the validation error can increase
-- before 
options.maxValidIncreasedEpochs = 5

----------------------------------------------------------------------
-- Initialize datasets
----------------------------------------------------------------------

local msds = require './importMSDS'

-- local _, unSets = ts_init.import_data(baseDir, setList, options)
local filter_suffix = '.h5'
local unSets = import_dataset.import_sets_filenames(msds.subset.path,
						    msds.subset.sets,
						    filter_suffix)

----------------------------------------------------------------------
-- Iterating over all potential models
----------------------------------------------------------------------

-- To save trained models for future use
-- (format: year_month_day-hour_minute_second)
local session_date = os.date('%y_%m_%d-%H_%M_%S', os.time())

-- modelsList = {modelMLP, modelCNN, modelInception, modelVGG};
modelsList = {modelLSTM};

-- Iterate over all models that we want to test
for k, v in ipairs(modelsList) do
   print('Start loop on models')
   -- Current model to check
   curModel = v();
   -- Define baseline parameters
   curModel:parametersDefault();
   ----------------------------------------------------------------------
   -- Structure optimization code
   ----------------------------------------------------------------------
   -- Define structure
   structure = {};
   -- Default initialization
   -- structure.nLayers = 3;
   -- structure.nInputs = options.resampleVal;
   -- structure.layers = {1000, 1000, 500, 200};
   -- structure.nOutputs = 10;
   -- structure.convSize = {16, 32, 64};
   -- structure.kernelWidth = {8, 8, 8};
   -- structure.poolSize = {2, 2, 2};
   -- structure.nClassLayers = 3;
   
   structure.nLayers = 1;
   structure.nInputs = options.resampleVal;
   structure.layers = {12, 1000};
   structure.nOutputs = structure.nInputs;
   structure.convSize = {16};
   structure.kernelWidth = {8};
   structure.poolSize = {2};
   structure.nClassLayers = 1;
   
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
   -- Define the classification criterion
   model, criterion = curModel:defineCriterion(model);
   
   -- Flatten all trainable parameters into a 1-dim vector
   if model then
      parameters, gradParameters = curModel:getParameters(model);
   end
   
   -- TODO: Temporary model redefinition
   model, criterion = nil
   model = nn.SeqLSTM(12, 12)
   criterion = nn.MSECriterion()
   criterion = nn.SequencerCriterion(criterion)
   
   local modelName = tostring(model):gsub('%.', '_')
   
   -- Set of trained layers
   trainedLayers = {};
   
   -- Unsupervised set
   local unsupData = unSets["TRAIN"];
   local unsupValid = unSets["VALID"];
   
   -- Switch training data to GPU
   if options.cuda then
      unsupData.data = unsupData.data:cuda();
      unsupValid.data = unsupValid.data:cuda();
   end
   
   for l = 1, structure.nLayers do
      print('Start loop on layers')
      -- To save intermediate results during learning
      local saveLocation = saveFolder .. 'results/' .. session_date ..
	 '-model-' .. modelName .. '-pretrain-layer' .. l .. '.net'
      
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
      
      local datasetEpoch
      -- Perform SGD on this subset of the dataset
      for datasetEpoch = 1, options.datasetMaxEpochs do
	 local minValidErr = math.huge
	 
	 local f_load = msds.load.get_btchromas
	 -- Perform sliding window over the whole dataset (too large to fit in memory)
	 -- Returns batches of training, validation... data as filenames
	 for slices in import_dataset.get_sliding_window_iterator(
	    unSets, f_load, options) do
	    -- print('Start loop on dataset windows')	 
	    local prevValid = math.huge
	    local validIncreasedEpochs = 0

	    print(minValidErr)
	    
	    local unsupValid = slices['VALID']

	    for epoch=0, options.maxEpochs do
	       print("Epoch #" .. epoch);

	       -- Create minibatch
	       local unsupData = {}
	       -- Gets random indexes for both inputs and targets
	       local indexes = torch.randperm(slices['TRAIN']['data']:size(1)):
		  sub(1, options.batchSize):long()
	       
	       for dataType, dataSubset in pairs(slices['TRAIN']) do
		  -- Iterate over inputs and targets
		  unsupData[dataType] = dataSubset:index(1, indexes)
	       end
	       
	       if (not options.adaptiveLearning) then
		  if torch.type(unsupData.data) ~= 'table' then
		     configureOptimizer(options, unsupData.data:size(2))
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
		     configureOptimizer(options, unsupData.data:size(2))
		  end
	       end
	       
	       --[[ Unsupervised pre-training ]]--
	       -- Perform unsupervised training of the model
	       err = curModel:unsupervisedTrain(model, unsupData, options);
	       print("Reconstruction error (train) : " .. err);
	       
	       --[[ Validation set checking ]]--
	       if epoch % options.validationRate == 0 then
		  -- Check reconstruction error on the validation data
		  validErr = curModel:unsupervisedTest(model, unsupValid,
						       options);
		  print("Reconstruction error (valid) : " .. validErr);
		  -- The validation error has risen since last checkpoint
		  if validErr > prevValid then
		     validIncreasedEpochs = validIncreasedEpochs + 1
		     print('Validation increased, now ' .. validIncreasedEpochs ..
			      ' times in a row')
		     if validIncreasedEpochs > options.maxValidIncreasedEpochs then
			-- Reload the last saved model
			model = torch.load(saveLocation);
			-- Stop the learning
			print(" => Stop learning");
			break;
		     end
		  else
		     validIncreasedEpochs = 0
		  end
		  print(validErr)
		  print(minValidErr)
		  if validErr <= minValidErr then
		     -- Save the current best model
		     -- TODO: check this! Since validation window is sliding,
		     -- this can break
		     torch.save(saveLocation, model);
		  end
		  -- Keep the current error
		  prevValid = validErr;
		  minValidErr = math.min(validErr, minValidErr)
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
