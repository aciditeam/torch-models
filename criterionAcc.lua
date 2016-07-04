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
-- accuracy measure (true positives over (total number of returned
-- positives + false negatives)).

-- Operates on batches of mutidimensional vectors.
-- Decorate with rnn.SequencerCriterion()

local binaryAccCriterion, parent = torch.class('nn.binaryAccCriterion',
					       'nn.Criterion')

function binaryAccCriterion:__init(threshold, regressionCriterion, sizeAverage)
   parent.__init(self)
   self.threshold = threshold or 0.2
   
   self.criterion = regressionCriterion or nn.AbsCriterion
   self.criterion = self.criterion()
   
   if sizeAverage ~= nil then
      self.criterion.sizeAverage = sizeAverage
   else
      self.criterion.sizeAverage = true
   end
   
   self.output_tensor = torch.zeros(1)
   self.total_weight_tensor = torch.ones(1)
   self.target = torch.zeros(1):long()
end

-- Thresholding function
-- 
-- Input: A batch of real multivariate points
-- Return: A batch of multivariate binary points
function binaryAccCriterion:tobinary(tensor)
   return tensor:gt(self.threshold):float()
end

-- Binarize tensors, compare inputs and targets and return:
-- TP / (TP + FP + FN)
-- where:
--  * TP is the number of True Positives (input and target are active)
--  * FP is the number of False Positives (input is active, target is not)
--  * FP is the number of False Negatives (target is active, input is not)
-- 
-- TODO: CHECK, this is actually not what's done, rather a standard L1 criterion
-- on the thresholded inputs (allows reusing nn.AbsCriterion's implementation)
function binaryAccCriterion:updateOutput(input, target)
   local classInput = self:tobinary(input)
   local classTarget = self:tobinary(target)
   output = self.criterion:forward(classInput, classTarget)
   
   self.output = output
   return self.output
end

function binaryAccCriterion:updateGradInput(input, target)
   local classInput = self:tobinary(input)
   local classTarget = self:tobinary(target)
   gradInput = self.criterion:backward(classInput, classTarget)

   self.gradInput = gradInput
   return self.gradInput
end
