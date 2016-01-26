----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for classification
-- Gated Recurrent Units (GRU) model
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'nngraph'
require 'unsup'
require 'optim'
require 'torch'
require 'rnn'

modelGRU = {};

----------------------------------------------------------------------
-- Handmade GRU
-- Creates one timestep of one GRU
-- Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
----------------------------------------------------------------------
function defineGRULayer(input_size, rnn_size, params)
  local n = params.n or 1
  local dropout = params.dropout or 0 
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  inputs[1] = nn.Identity()() -- x
  for L = 1,n do
    inputs[L+1] = nn.Identity()() -- prev_h[L]
  end
  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do

    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]; --OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    print(x);
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

module = nn.Sequencer(defineGRULayer(1, 2, {n = 2, dropout = 0.5}));
x=torch.rand(1, 128);
y=torch.rand(128, 100);
sX = x[{{}, {1,10}}];
print(torch.cat(sX,sX));
print(module:forward({{sX,torch.cat(x:t(),x:t()),torch.cat(x:t(),x:t())}}))