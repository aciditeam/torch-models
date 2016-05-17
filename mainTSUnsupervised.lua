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
require 'datasets/importTSDataset'
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

----------------------------------------------------------------------
-- Initialize datasets
----------------------------------------------------------------------

local _, unSets = ts_init.import_data(baseDir, setList, options)

----------------------------------------------------------------------
-- Iterating over all potential models
----------------------------------------------------------------------

-- modelsList = {modelMLP, modelCNN, modelInception, modelVGG};
modelsList = {modelMLP};

-- Iterate over all models that we want to test
for k, v in ipairs(modelsList) do
   -- Current model to check
   curModel = v();
   -- Define baseline parameters
   curModel:parametersDefault();
   ----------------------------------------------------------------------
   -- Structure optimization code
   ----------------------------------------------------------------------
   -- Define structure
   structure = {};
   structure.nLayers = 3;
   structure.nInputs = options.resampleVal;
   structure.layers = {1000, 1000, 500, 200};
   structure.nOutputs = 10;
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

   -- Unsupervised set
   local unsupData = unSets["TRAIN"];
   local unsupValid = unSets["VALID"];

   -- Switch training data to GPU
   if options.cuda then
      unsupData.data = unsupData.data:cuda();
      unsupValid.data = unsupValid.data:cuda();
   end

   -- Set of trained layers
   trainedLayers = {};

   for l = 1,structure.nLayers do
      -- Define the pre-training model
      local model = curModel:definePretraining(structure, l, options);
      print(tostring(model));
      -- Activate CUDA on the model
      if options.cuda then model:cuda(); end
      -- If classical learning configure the optimizer
      if (not options.adaptiveLearning) then 
	 if torch.type(unsupData.data) ~= 'table' then
	    configureOptimizer(options, unsupData.data:size(2))
	 else
	    configureOptimizer(options, #unsupData.data);
	 end   
      end

      epoch = 0;
      prevValid = math.huge;

      while epoch < options.maxEpochs do
	 print("Epoch #" .. epoch);
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
		  unsupData.data:size(2));
	    end
	    -- Remaining epochs = Advanced learning algorithm user-selected
	    -- (LBFGS | CG | ADADELTA | ADAGRAD | ADAM | ADAMAX | FISTALS | NAG | RMSPROP | RPROP | CMAES)
	    if epoch == options.superLinearEpoch then configureOptimizer(options, unsupData.data:size(2)); end
	 end
	 
	 --[[ Unsupervised pre-training ]]--
	 -- Perform unsupervised training of the model
	 error = curModel:unsupervisedTrain(model, unsupData, options);
	 print("Reconstruction error (train) : " .. error);

	 --[[ Validation set checking ]]--
	 if epoch % options.validationRate == 0 then
	    -- Check reconstruction error on the validation data
	    validError = curModel:unsupervisedTest(model, unsupValid,
						   options);
	    print("Reconstruction error (valid) : " .. validError);
	    -- The validation error has risen since last checkpoint
	    if validError > prevValid then
	       -- Reload the last saved model
	       torch.load('results/model-pretrain-layer' .. l .. '.net');
	       -- Stop the learning
	       print(" => Stop learning");
	       break; 
	    end
	    -- Otherwise save the current model
	    torch.save('results/model-pretrain-layer' .. l .. '.net',
		       model);
	    -- Keep the current error
	    prevValid = validError;
	 end
	 epoch = epoch + 1;
	 -- Collect the garbage
	 collectgarbage();
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
