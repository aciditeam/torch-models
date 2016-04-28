----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

-- Define the optimization structure and main parameters
function setDefaultConfiguration()
  local options = {};
  -- TODO
  -- TODO
  -- Separate the different parameters between
  --  * default
  --  * distribution
  --  * values
  --  * model
  --  * learning
  --  * global
  -- TODO
  -- TODO
  ----------------------------------------------------------------------
  -- Global options
  ----------------------------------------------------------------------
  options.save                  = 'results/';   -- subdirectory to save/log experiments in
  options.statinterval          = 5000          -- interval for saving stats and models
  options.verbose               = false         -- be verbose
  options.display               = true          -- display stuff
  options.visualize             = false;        -- visualize input data and weights during training
  options.plot                  = false;        -- live plot
  ----------------------------------------------------------------------
  -- Learning options
  ----------------------------------------------------------------------
  options.optimization          = 'ADAM';       -- optimization method: SGD | ASGD | CG | LBFGS
  options.lambda                = 0.2           -- sparsity coefficient
  options.beta                  = 0.1           -- prediction error coefficient
  options.eta                   = 2e-3          -- learning rate
  options.etadecay              = 1e-3          -- learning rate decay
  options.learningRate          = 1e-5;         -- learning rate at t=0
  options.batchSize             = 32;           -- mini-batch size (1 = pure stochastic)'
  options.weightDecay           = 0.2;          -- weight decay (SGD only)
  options.momentum              = 0.8;          -- gradient momentum (SGD only)
  options.t0                    = 1;            -- start averaging at t0 (ASGD only), in nb of epochs
  options.maxIter               = 1e9;          -- maximum nb of pre-training iterations
  options.maxEpochs             = 1000;         -- maximum nb of epochs for learning
  options.nbIter                = 1000000       -- max number of updates
  options.type                  = 'float'       -- type of the data: float|double|cuda
  -- for linear model only:
  options.tied                  = false         -- decoder weights are tied to encoder weights (transposed)
  -- use hessian information for training:
  options.hessian               = false         -- compute diagonal hessian coefficients to condition learning rates
  options.hessiansamples        = 500           -- number of samples to use to estimate hessian
  options.hessianinterval       = 10000         -- compute diagonal hessian coefs at every this many samples
  options.minhessian            = 0.02          -- min hessian to avoid extreme speed up
  options.maxhessian            = 500           -- max hessian to avoid extreme slow down
  ----------------------------------------------------------------------
  -- Models options
  ----------------------------------------------------------------------
  options.model                 = 'linear'      -- auto-encoder class: linear | linear-psd | conv | conv-psd
  options.pretrain              = 0
  options.kernelsize            = 9             -- size of convolutional kernels
  options.filtersin             = 1             -- number of input convolutional filters
  options.filtersout            = 16            -- number of output convolutional filters
  ----------------------------------------------------------------------
  -- Optimization and regularization
  ----------------------------------------------------------------------
  options.validPercent          = 0.1           -- percentage of dataset to use as validation  
  options.resampleVal           = 128           -- size of time series resampling 
  options.subLinearEpoch        = 3             -- At which epoch we should start mini-batch sub-linear SGD
  options.superLinearEpoch      = 10            -- At which epoch we should start super-linear higher-level algorithm
  options.regularizeL1          = 0
  options.regularizeL2          = 0           
  options.maxValidRise          = 3
  options.validationRate        = 5
  options.adaptiveLearning      = false             
  options.zcaWhitening          = false
  options.gcnNormalize          = false
  options.dataAugmentation      = true
  
  local distributions = {};
  ----------------------------------------------------------------------
  -- Distributions
  ----------------------------------------------------------------------
  distributions.optimization          = {'SGD','ASGD','LBFGS','CG','ADADELTA','ADAGRAD','ADAM','ADAMAX','FISTALS','NAG','RMSPROP','RPROP','CMAES'};
  distributions.lambda                = {uniform, 0, 1}             -- sparsity coefficient
  distributions.beta                  = {uniform, 0, 1}             -- prediction error coefficient
  distributions.eta                   = {normal, 1e-1, 1e-1}             -- learning rate
  distributions.etadecay              = {normal, 1e-4, 1e-4}          -- learning rate decay
  --
  -- TODO
  -- TODO
  -- Finish the distributions
  -- + Finish hyperparameters optimization
  -- TODO
  -- TODO
  --
  return options, distributions
end