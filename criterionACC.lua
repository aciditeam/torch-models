----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Frame-level accuracy criterion for finite-domain symbolic sequences
--
-- As presented in Bay et al., ISMIR2009:
-- Evaluation of multiple F0-estimation and tracking systems
-- 
-- Used in Boulanger-Lewandowski et al., ICML2012:
-- Modeling Temporal Dependencies in High-Dimensional Sequences
-- 
----------------------------------------------------------------------

-- This criterion uses a threshold to turn a multivariate time-series
-- into a multidimensional binary time-series and perform a standard
-- accuracy measure (true positives over total number of returned
-- positives).

-- Operates on batches of mutidimensional vectors.
-- Used by decorating it with an rnn.SequencerCriterion()

-- Use ClassNLLCriterion and see the 12-dimensional vectors as binary
-- encodings of integers => a LOT of classes (2^12-1).
-- TODO, CHECK: is this a good idea?

local output

local gradInput

local threshold

local classificationCriterion

local function binarydecode(tensor)
   local powersof2 = torch.cpow(torch.range(0, tensor:size(2)), 2)

   return torch.mv(tensor, powersof2)
end

local function tobinary(tensor)
   return tensor:gt(threshold)
end

local function toclass(...) = binarydecode(tobinary(...))

function forward(input, target)
   local classInput = toclass(input)
   local classTarget = toclass(target)
   classificationCriterion:forward(classInput, classTarget)
end

