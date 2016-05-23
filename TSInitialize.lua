----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Initialization functions
--
----------------------------------------------------------------------


require './mainParameters'
local ts_import = require './importTSDataset'
local preprocess = require './mainPreprocess'


local M = {}

----------------------------------------------------------------------
-- Define the global parameters
----------------------------------------------------------------------

-- Initialize default set of options
--
-- Input:
--  * use_cuda, a boolean: whether to enable CUDA processing
function M.get_options(use_cuda)
   -- Create a default configuration
   options = setDefaultConfiguration();
   -- Override some parameters
   options.visualize = true;
   if use_cuda then
      options.cuda = true;
   end
   return options
end

----------------------------------------------------------------------
-- Global variables and CUDA handling
----------------------------------------------------------------------

-- Eventual CUDA support
function M.set_cuda(options)
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
end

-- Global parameters
function M.set_globals()
   -- Switching to float (economic)
   torch.setdefaulttensortype('torch.FloatTensor')
   -- Multi-threading
   torch.setnumthreads(4);
end

return M
