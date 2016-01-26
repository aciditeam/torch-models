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
require 'importTSDataset'
require 'mainLearning'
require 'mainParameters'
require 'mainProfiler'
require 'mainUtils'
require 'modelCriterias'
require 'modelCNN'
require 'modelDBM'
require 'modelLSTM'
require 'modelMLP'
require 'modelRNN'
require 'modelVGG'
require 'modelInception'
require 'modelNIN'
require 'modelAE'

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
options.cuda = false;
if options.cuda then
   print('==> switching to CUDA')
   require 'cunn'
end
-- Multi-threading
torch.setnumthreads(4);

----------------------------------------------------------------------
-- Datasets variables
----------------------------------------------------------------------

-- Change this directory to point on all UCR datasets
baseDir = '/Users/esling/Dropbox/TS_Datasets';

--[[
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
]]--

setList = {'ArrowHead','Beef','BeetleFly','BirdChicken'};

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
print " - Constructing unsupervised superset"
-- Also prepare a very large unsupervised dataset
unSets = construct_unsupervised(sets);
-- TODO 
-- Also needs a validation subset for pre-training !
-- TODO
print " - Constructing balanced validation subsets"
-- We shall construct a balanced train/validation subset
sets = construct_validation(sets, options.validPercent);

----------------------------------------------------------------------
-- Additional pre-processing code
----------------------------------------------------------------------

-- TODO
-- TODO
-- Pre-condition the inputs (Everything should look like a circle)
-- Normalize in similar ranges
--  - ZCA Whitening
--  - Other pre-processing techniques ?
-- TODO
-- TODO

----------------------------------------------------------------------
-- Data augmentation code
----------------------------------------------------------------------

-- TODO
-- TODO
-- Perform data augmentation
-- Time series case :
--  * Noise, outlier, warp
--  * Eventually crop, scale (sub-seq)
-- cf. Manifold densification (Matlab)
-- TODO
-- TODO

----------------------------------------------------------------------
-- Iterating over all potential models
----------------------------------------------------------------------

-- modelsList = {modelMLP, modelCNN, modelInception, modelVGG};
modelsList = {modelLSTM};

-- Iterate over all models that we want to test
for k, v in ipairs(modelsList) do
  -- Current model to check
  curModel = v();
  print(curModel);
  -- Define baseline parameters
  curModel:parametersDefault();
  print(curModel);
  ----------------------------------------------------------------------
  -- Structure optimization code
  ----------------------------------------------------------------------
  -- Define structure
  structure = {};
  --[[
  structure.nLayers = 6;
  structure.nInputs = options.resampleVal;
  structure.layers = {500, 500, 250};
  structure.nOutputs = 10;
  structure.convSize = {16, 32, 64, 32, 32, 32};
  structure.kernelWidth = {8, 8, 8, 8, 4, 4};
  structure.poolSize = {2, 2, 2, 2, 1, 1};
  structure.nClassLayers = 3;
  ]]--
  structure.nLayers = 3;
  structure.nInputs = options.resampleVal;
  structure.layers = {500, 500, 250};
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
    unsupData = unSets;
    -- Set of trained layers
    trainedLayers = {};
    for l = 1,structure.nLayers do
      -- Prepare the layer properties
      if l == 1 then inS = options.resampleVal; else inS = structure.layers[l - 1]; end
      outS = structure.layers[l]; 
      -- Define the pre-training model
      local model = curModel:definePretraining(inS, outS, options);
      -- Activate CUDA on the model
      if options.cuda then model:cuda(); end
      epoch = 0;
      while epoch < options.maxEpochs do
        print("Epoch #" .. epoch);
        print(unsupData.data:size(2));
        --[[ Adaptive learning ]]--
        -- 1st epochs = Start with purely stochastic (SGD) on single examples
        if epoch == 0 then configureOptimizer({optimization = 'SGD', batchSize = 1, learningRate = 5e-3}, unsupData.data:size(2)); end
        -- Next epochs = Sub-linear approximate algorithm ASGD with mini-batches
        if epoch == options.subLinearEpoch then configureOptimizer({optimization = 'SGD', batchSize = 128, learningRate = 2e-3}, unsupData.data:size(2)); end
        -- Remaining epochs = Advanced learning algorithm user-selected (LBFGS | CG | ADADELTA | ADAGRAD | ADAM | ADAMAX | FISTALS | NAG | RMSPROP | RPROP | CMAES)
        if epoch == options.superLinearEpoch then configureOptimizer(options, unsupData.data:size(2)); end
        -- Perform unsupervised training of the model
        error = curModel:unsupervisedTrain(model, unsupData, options);
        print("Reconstruction error : " .. error);
        epoch = epoch + 1;
        -- TODO
        -- NEEDS A VALIDATION SUBSET AND TO CHECK ;
        --[[
        validError = curModel:supervisedTest(model, sets[value]["VALID"], options);
        confusion:zero();
        -- Add a new iteration of increase in validation error
        if validError > prevValid then validRise = validRise + 1; else validRise = 0 end
        -- Check if we need to break the learning
        if validRise >= curModel.maxValidRise then break; end
        prevValid = validError;
        ]]--
        -- TODO
      end
      -- Keep trained layer in table
      trainedLayers[l] = model;
      -- Prepare a set of activations
      forwardedData = {data = torch.Tensor(unsupData.data:size(1), outS)};
      -- Retrieve the encoding layer only
      model = curModel:retrieveEncodingLayer(model)
      -- Perform forward propagation
      for t = 1,unsupData.data:size(1) do
        input = unsupData.data[t];
        forwardedData.data[t] = model:forward(input)
      end
      -- Replace previous set
      unsupData = forwardedData;
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
    model = curModel:defineModel(structure, option);
    -- Initialize weights
    if curModel.pretraining then
      model = curModel:weightsTransfer(model, trainedLayers);
    else
      model = curModel:weightsInitialize(model);
    end
    -- Check the model
    print(model);
    -- Define the classification criterion
    model, criterion = defineCriterion(model);
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
    if model then
      parameters,gradParameters = model:getParameters();
    end
    epoch = 0;
    validRise = 0;
    prevValid = 1.0;
    options.learningRate = 1e-4;
    configureOptimizer(options, inSize);
    while epoch < options.maxEpochs do
      --[[ Adaptive learning ]]--
      -- 1st epochs = Start with purely stochastic (SGD) on single examples
      --if epoch == 0 then configureOptimizer({optimization = 'SGD', batchSize = 1, learningRate = 1}, sets[value]["TRAIN"].data:size(2)); end
      -- Next epochs = Sub-linear approximate algorithm ASGD with mini-batches
      --if epoch == options.subLinearEpoch then configureOptimizer({optimization = 'SGD', batchSize = 128, learningRate = 0.1}, sets[value]["TRAIN"].data:size(2)); end
      -- Remaining epochs = Advanced learning algorithm user-selected (LBFGS | CG | ADADELTA | ADAGRAD | ADAM | ADAMAX | FISTALS | NAG | RMSPROP | RPROP | CMAES)
      --if epoch == options.superLinearEpoch then configureOptimizer(options, inSize); end
      -- TODO
      -- We will use different learning rates for different layers (based on average size of gradients and weights)
      -- learningRate = 0.01 * (avgWeight / avgGradient);
      -- Use a decaying learning rate (start very large, when validation stops improving, divide by 2)
      -- TODO
      
      if validRise > 2 then learningRate = learningRate * learningRateDecay; end
    
      --[[ Training data ]]--
      trainError = curModel:supervisedTrain(model, sets[value]["TRAIN"], options);
      confusion:zero();
      --[[ Validation testing ]]--
      validError = curModel:supervisedTest(model, sets[value]["VALID"], options);
      confusion:zero();
      -- Add a new iteration of increase in validation error
      if validError > prevValid then validRise = validRise + 1; else validRise = 0 end
      -- Check if we need to break the learning
      if validRise >= options.maxValidRise then break; end
      prevValid = validError;
      --[[ Test dataset evluation ]]--
      testError = curModel:supervisedTest(model, sets[value]["TEST"], options);
      confusion:zero();
    
      -- Periodically collect statistics and monitor
      -- Visualize all these
      print('Train error = ' .. trainError);
      print('Valid error = ' .. validError);
      print('Test error = ' .. testError);
    
    end
  end
end