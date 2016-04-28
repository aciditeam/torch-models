----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'rnn'
require 'unsup'
require 'optim'
require 'torch'
require 'modelLSTM'
require 'modelClass'
local nninit = require 'nninit'

local modelBLSTM, parent = torch.class('modelBLSTM', 'modelLSTM')

function modelBLSTM:defineModel(structure, options)
  -- Container
  local model = nn.Sequential();
  -- Hidden layers
  for i = 1,structure.nLayers do
    -- Long Short-Term Memories
    if i == 1 then
      if (self.sequencer) then
        curLSTM = nn.FastLSTM(self.windowSize, structure.layers[i], self.rho);
      else
        curLSTM = nn.FastLSTM(structure.nInputs, structure.layers[i], self.rho);
      end
    else
      curLSTM = nn.FastLSTM(structure.layers[i-1], structure.layers[i], self.rho);
    end
    -- Always initialize the bias of the LSTM forget gate to 1 (trick from old RNN study)
    if self.initForget then curLSTM.i2g.bias[{{2*structure.layers[i]+1,3*structure.layers[i]}}]:fill(1) end
    -- Add the bias-adjusted LSTM to the network
    model:add(curLSTM);
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
    lstmModel = nn.Sequencer(model);
    model = nn.Sequential();
    -- Number of windows we will consider
    local nWins = torch.ceil((structure.nInputs - self.windowSize + 1) / self.windowStep)
    -- Here we add the subsequencing trick
    model:add(nn.SlidingWindow(2, self.windowSize, self.windowStep));
    model:add(lstmModel);
    model:add(nn.JoinTable(2));
    model:add(nn.Linear(nWins * structure.nOutputs, structure.nOutputs));
  else
    -- Recursor case
    lstmLayers = nn.Recursor(model);
    model = nn.Sequential();
    -- Add the LSTM layers
    model:add(lstmLayers);
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