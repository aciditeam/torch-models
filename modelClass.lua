----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main definition of the model class
--   * Synthesize the basic functionalities of most models
--   * Allows to 
--   * Will define basic train / test functions
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
-- local class = require 'class'
require 'torch'
require 'mainLearning'

-- Defining the main model class
local modelClass = torch.class('modelClass');

--- This function defines the construction for the structure of a learning model
--
-- The code takes a particular network topology (given as a lua table)
-- with a set of options and outputs a model compatible with nn.Module 
-- 
-- @param   structure   Network topology (size) (Table)
-- @param   options     Sets of options         (Table)
-- @return  model       The constructed model   (nn.Module)
-- @usage   model = register_person('john','doe')
-- @see Person
-- 
function modelClass:defineModel(structure, options)
end

-- Set the default parameters
function modelClass:parametersDefault()
end

-- Set a set of parameters (coming from hyper-parameter optimization)
function modelClass:parametersSet(parameters)
end

-- Defines the structure of eventual pre-training model
function modelClass:definePretraining(structure, l, options)
end

-- Function to perform unsupervised training on a sub-model
function modelClass:unsupervisedTrain(model, unsupData, options)
  return unsupervisedTrain(model, unsupData, options);
end

-- Function to perform unsupervised testing on a sub-model
function modelClass:unsupervisedTest(model, data, options)
  return unsupervisedTest(model, data, options);
end

-- Function to perform supervised training on the full model
function modelClass:supervisedTrain(model, data, options)
  return supervisedTrain(model, data, options);
end

-- Function to perform supervised testing on the model
function modelClass:supervisedTest(model, data, options)
  return supervisedTest(model, data, options);
end

function modelClass:defineCriterion(model)
  model:add(nn.LogSoftMax());
  criterion = nn.ClassNLLCriterion();
  return model, criterion;
end

function modelClass:getParameters(model)
  return model:getParameters();
end

-- Transfer the weights from pre-training to the final model
function modelClass:weightsTransfer(model, trainedLayers)
end

function modelClass:weightsInitialize(model)
end