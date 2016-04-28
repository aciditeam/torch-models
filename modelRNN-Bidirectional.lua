----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Bi-directional RNN
-- Applies encapsulated fwd and bwd rnns to an input sequence in forward and reverse order.
-- brnn = nn.BiSequencer(fwd, [bwd, merge])
-- The input to the module is a sequence (a table) of tensors and the output is a sequence (a table) of tensors of the same length. 
-- Applies a fwd rnn (an AbstractRecurrent instance) to each element in the sequence in forward order and applies the bwd rnn in reverse order 
-- (from last element to first element). The bwd rnn defaults to:
-- bwd = fwd:clone()
-- bwd:reset()
-- For each step (in the original sequence), the outputs of both rnns are merged together using the merge module 
-- (defaults to nn.JoinTable(1,1)). If merge is a number, it specifies the JoinTable constructor's nInputDim argument. Such that the merge module is then initialized as :
-- merge = nn.JoinTable(1,merge)
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'rnn'
require 'unsup'
require 'optim'
require 'torch'
require 'modelRNN'
local nninit = require 'nninit'

modelBRNN = {};

local modelBRNN, parent = torch.class('modelBRNN', 'modelRNN')

function modelBRNN:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  -- Hidden layers
  for i = 1,structure.nLayers do
    if i == 1 then nIn = self.windowSize; else nIn = structure.layers[i - 1]; end
    -- Prepare one layer of reccurent computation
    local r = nn.Recurrent(
      structure.layers[i], 
      nn.Identity(),
      nn.Linear(nIn, structure.layers[i]), 
      self.nonLinearity(),
      self.rho
    );
    model:add(r);
    -- Layer-wise linear transform
    if self.layerwiseLinear then model:add(nn.Linear(structure.layers[i], structure.layers[i])) end
    -- Batch normalization
    if self.batchNormalize then model:add(nn.BatchNormalization(structure.layers[i])); end
    -- Non-linearity
    if self.addNonLinearity then model:add(self.nonLinearity()); end
    -- Dropout
    if self.dropout then model:add(nn.Dropout(self.dropout)); end
  end
  -- Final regression layer for classification
  if self.sequencer then 
    -- Sequencer case simply needs to add a linear transform to number of classes
    model:add(nn.Linear(structure.layers[structure.nLayers], structure.nOutputs))
    rnnModel = nn.Sequencer(model);
    model = nn.Sequential();
    -- Number of windows we will consider
    local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
    -- Here we add the subsequencing trick
    model:add(nn.SlidingWindow(2, self.windowSize, self.windowStep));
    model:add(rnnModel);
    model:add(nn.JoinTable(2));
    model:add(nn.Linear(nWins * structure.nOutputs, structure.nOutputs));
  else
    -- Recursor case
    rnnLayers = nn.Recursor(model, self.rho);
    model = nn.Sequential()
    -- Add the recurrent 
    model:add(rnnLayers);
    -- Needs to reshape the data from all outputs
    model:add(nn.Reshape(structure.layers[structure.nLayers]));
    -- And then add linear transform to number of classes
    model:add(nn.Linear(structure.layers[structure.nLayers], structure.nOutputs))
  end
  -- [[ Bi-directionnal aspect of the model ]]--
  local backward = model:clone();
  backward:reset();
  backward:remember('neither');
  local brnn = nn.BiSequencer(model, backward);
  -- TODO
  -- Maybe need to add some pre-linearization
  -- + Not sure what the output is going to be
  -- TODO
  return nn.Sequential():add(brnn);
end
