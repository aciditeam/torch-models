----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
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
-- Define the global parameters
----------------------------------------------------------------------

-- Create a default configuration
options = setDefaultConfiguration();
-- Override some parameters
options.visualize = true;

----------------------------------------------------------------------
-- Global variables and CUDA handling
----------------------------------------------------------------------

-- Switching to float (economic)
torch.setdefaulttensortype('torch.FloatTensor')
-- Eventual CUDA support
options.cuda = true;
if options.cuda then
   print('==> switching to CUDA')
   local ok, cunn = pcall(require, 'fbcunn')
   if not ok then ok, cunn = pcall(require,'cunn') end
   if not ok then
      print("Impossible to load CUDA (either fbcunn or cunn)"); os.exit()
   end
   local ok, cunn = pcall(require, 'fbcunn')
   deviceParams = cutorch.getDeviceProperties(1)
   cudaComputeCapability = deviceParams.major + deviceParams.minor / 10
end
-- Multi-threading
torch.setnumthreads(4);

----------------------------------------------------------------------
-- Datasets variables
----------------------------------------------------------------------

-- Change this directory to point on all UCR datasets
baseDir = '/home/aciditeam/datasets/TS_Datasets';

setList = {'50words','Adiac','ArrowHead','ARSim','Beef',
	   'BeetleFly','BirdChicken','Car','CBF','Chlorine','CinECG',
	   'Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSize',
	   'DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect',
	   'DistalPhalanxTW','Earthquakes','ECG200','ECG5000','ECGFiveDays',
	   'ElectricDevices','FaceAll','FaceFour','FacesUCR','Fish','FordA','FordB',
	   'Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate',
	   'InsectWingbeatSound','Ionosphere','ItalyPower','LargeKitchenAppliances',
	   'Lighting2','Lighting7','MALLAT','Meat','MedicalImages',
	   'MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect',
	   'MiddlePhalanxTW','MoteStrain','NonInv_ECG1','NonInv_ECG2','OliveOil',
	   'OSULeaf','PhalangesOutlinesCorrect','Phoneme','Plane',
	   'ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect',
	   'ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim',
	   'ShapesAll','SmallKitchenAppliances','Sonar','SonyAIBO1','SonyAIBO2',
	   'StarLight','Strawberry','SwedishLeaf','Symbols','Synthetic',
	   'ToeSegmentation1','ToeSegmentation2','Trace','Two_Patterns','TwoLeadECG',
	   'uWGestureX','uWGestureY','uWGestureZ','Vehicle','Vowel','Wafer','Waveform','Wdbc','Wine',
	   'Wins','WordsSynonyms','Worms','WormsTwoClass','Yeast','Yoga'};

--setList = {'ArrowHead','Beef','BeetleFly','BirdChicken'};
--setList = {'50words','Adiac','ArrowHead','ARSim','Beef','BeetleFly','BirdChicken'};

----------------------------------------------------------------------
-- Import datasets code
----------------------------------------------------------------------

print " - Importing datasets";
local sets = import_data(baseDir, setList, options.resampleVal);
print " - Checking data statistics";
for key,value in ipairs(setList) do
   v = sets[value]["TRAIN"];
   meanData = v.data[{{},{}}]:mean();
   stdData = v.data[{{},{}}]:std();
   print('    - '..value..' [TRAIN] - mean: ' .. meanData .. ', standard deviation: ' .. stdData);
   v = sets[value]["TEST"];
   meanData = v.data[{{},{}}]:mean();
   stdData = v.data[{{},{}}]:std();
   print('    - '..value..' [TEST] - mean: ' .. meanData .. ', standard deviation: ' .. stdData);
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

-- Global Contrast Normalization
if options.gcnNormalize then
   print ' - Perform Global Contrast Normalization (GCN) on input data'
   require 'mainPreprocess'
   unSets["TRAIN"].data = gcn(unSets["TRAIN"].data);
   unSets["VALID"].data = gcn(unSets["VALID"].data);
   for k, v in ipairs(setList) do
      sets[v]["TRAIN"].data = gcn(sets[v]["TRAIN"].data);
      sets[v]["VALID"].data = gcn(sets[v]["VALID"].data);
      sets[v]["TEST"].data = gcn(sets[v]["TEST"].data);
   end
end
-- Zero-Component Analysis whitening
if options.zcaWhitening then
   print ' - Perform Zero-Component Analysis (ZCA) whitening'
   require 'mainPreprocess'
   local means, P = zca_whiten_fit(
      torch.cat(unSets["TRAIN"].data, unSets["VALID"].data, 1));
   unSets["TRAIN"].data = zca_whiten_apply(unSets["TRAIN"].data, means, P)
   unSets["VALID"].data = zca_whiten_apply(unSets["VALID"].data, means, P)
   for k, v in pairs(setList) do
      local means, P = zca_whiten_fit(
	 torch.cat(sets[v]["TRAIN"].data, sets[v]["VALID"].data, 1));
      sets[v]["TRAIN"].data = zca_whiten_apply(sets[v]["TRAIN"].data, means, P)
      sets[v]["VALID"].data = zca_whiten_apply(sets[v]["VALID"].data, means, P)
      sets[v]["TEST"].data = zca_whiten_apply(sets[v]["TEST"].data, means, P)
   end
end

----------------------------------------------------------------------
-- Iterating over all potential models
----------------------------------------------------------------------

-- TODO
-- cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
-- cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
-- cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
-- cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
-- TODO

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
   if curModel.pretrain then
      print('- Performing pretraining.');
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
	 prevValid = 5e20;
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
   ----------------------------------------------------------------------
   -- Supervised classification code
   ----------------------------------------------------------------------
   -- Evaluate over all datasets
   for key,value in ipairs(setList) do
      print("    * (MLP) Classifying " .. value);
      -- Data input size
      inSize = sets[value]["TRAIN"].data:size(2);
      -- Retrieve set of unique classes
      classes = uniqueTensor(sets[value]["TRAIN"].labels);
      -- Change the number of last layer units
      structure.nOutputs = #classes;
      -- Define the model
      model = curModel:defineModel(structure, options);
      -- Initialize weights
      if curModel.pretrain then
	 model = curModel:weightsTransfer(model, trainedLayers);
      else
	 model = curModel:weightsInitialize(model);
      end
      -- Check the model
      print(tostring(model));
      -- Define the classification criterion
      model, criterion = curModel:defineCriterion(model);
      -- TODO
      -- TODO
      -- Sub-loop on hyper-parameter optimization !
      -- Also for criterion !
      -- TODO
      -- TODO
      -- Eventual CUDA support
      if options.cuda then
	 model:cuda();
	 criterion:cuda();
      end
      -- This matrix records the current confusion across classes
      confusion = optim.ConfusionMatrix(classes);
      -- Log results to files
      trainLogger = optim.Logger(paths.concat(options.save, 'train.log'));
      testLogger = optim.Logger(paths.concat(options.save, 'test.log'));
      -- TODO
      -- TODO
      -- Needs more logging / monitoring
      -- cf. Separate visualize file
      -- TODO
      -- TODO
      -- Flatten all trainable parameters into a 1-dim vector
      if model then parameters, gradParameters = curModel:getParameters(model); end
      epoch = 0;
      validRise = 0;
      prevValid = 1.0;
      options.learningRate = 1e-4;
      configureOptimizer(options, inSize);
      while epoch < options.maxEpochs do
	 --[[ Adaptive learning ]]--
	 if options.adaptiveLearning then
	    -- 1st epochs = Start with purely stochastic (SGD) on single examples
	    if epoch == 0 then configureOptimizer({optimization = 'SGD', batchSize = 1, learningRate = 5e-3}, sets[value]["TRAIN"].data:size(2)); end
	    -- Next epochs = Sub-linear approximate algorithm ASGD with mini-batches
	    if epoch == options.subLinearEpoch then
	       configureOptimizer({optimization = 'SGD', batchSize = 128,
				   learningRate = 2e-3},
		  sets[value]["TRAIN"].data:size(2));
	    end
	    -- Remaining epochs = Advanced learning algorithm user-selected (LBFGS | CG | ADADELTA | ADAGRAD | ADAM | ADAMAX | FISTALS | NAG | RMSPROP | RPROP | CMAES)
	    if epoch == options.superLinearEpoch then
	       configureOptimizer(options, inSize)
	    end
	    -- We will use different learning rates for different layers (based on average size of gradients and weights)
	    -- learningRate = 0.01 * (avgWeight / avgGradient);
	    -- Use a decaying learning rate (When validation error rise, divide by decay)
	    if validRise > 2 then
	       learningRate = learningRate * learningRateDecay
	    end
	 end
	 --[[ Training data ]]--
	 trainError = curModel:supervisedTrain(model, sets[value]["TRAIN"],
					       options);
	 confusion:zero();
	 --[[ Validation testing ]]--
	 validError = curModel:supervisedTest(model, sets[value]["VALID"],
					      options);
	 confusion:zero();
	 -- Add a new iteration of increase in validation error
	 if validError > prevValid then
	    validRise = validRise + 1;
	 else
	    validRise = 0
	 end
	 -- Check if we need to break the learning
	 if validRise >= options.maxValidRise then break; end
	 prevValid = validError;
	 --[[ Test dataset evluation ]]--
	 testError = curModel:supervisedTest(model, sets[value]["TEST"],
					     options);
	 confusion:zero();
	 
	 -- Periodically collect statistics and monitor
	 -- Visualize all these
	 print('Train error = ' .. trainError);
	 print('Valid error = ' .. validError);
	 print('Test error = ' .. testError);
	 
      end
   end
end
