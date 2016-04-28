----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Utility functions
--
----------------------------------------------------------------------

function uniqueTable(table)
  local hash = {}
  local res = {}
  for _,v in ipairs(table) do
    if (not hash[v]) then
      res[#res+1] = v -- you could print here instead of saving to result table if you wanted
      hash[v] = true
    end
  end
  return res;
end

function uniqueTensor(table)
  local hash = {}
  local res = {}
  for i = 1,table:size(1) do
    if (not hash[table[i]]) then
      res[#res+1] = table[i] -- you could print here instead of saving to result table if you wanted
      hash[table[i]] = true
    end
  end
  return res;
end

function gradientCheck()
  local decGP_est, encGP_est = torch.DoubleTensor(decGradParams:size()), torch.DoubleTensor(encGradParams:size())

  -- Easy function to do forward pass over coupled network and get error
  function forwardPass()
    local encOut = enc:forward(encInSeq)
    forwardConnect(encLSTM, decLSTM)
    local decOut = dec:forward(decInSeq)
    local E = criterion:forward(decOut, decOutSeq)
    return E
  end

  -- Check encoder
  for i = 1, encGradParams:size(1) do
    -- Forward with \theta+eps
    encParams[i] = encParams[i] + eps
    local C1 = forwardPass()
    -- Forward with \theta-eps
    encParams[i] = encParams[i] - 2 * eps
    local C2 = forwardPass()

    encParams[i] = encParams[i] + eps
    encGP_est[i] = (C1 - C2) / (2 * eps)
  end
  tester:assertTensorEq(encGradParams, encGP_est, eps, "Numerical gradient check for encoder failed")

  -- Check decoder
  for i = 1, decGradParams:size(1) do
    -- Forward with \theta+eps
    decParams[i] = decParams[i] + eps
    local C1 = forwardPass()
    -- Forward with \theta-eps
    decParams[i] = decParams[i] - 2 * eps
    local C2 = forwardPass()
    decParams[i] = decParams[i] + eps
    decGP_est[i] = (C1 - C2) / (2 * eps)
  end
  tester:assertTensorEq(decGradParams, decGP_est, eps, "Numerical gradient check for decoder failed")
end
