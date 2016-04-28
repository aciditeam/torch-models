----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Defining various criterions
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'torch'
require 'nn'

-- Deprecated
function defineCriterion(model)
  model:add(nn.LogSoftMax());
  criterion = nn.ClassNLLCriterion();
  return model, criterion;
end

----------------------------------------------------------------------
-- Absolute Criterion
-- Measures the mean absolute value of the element-wise difference between input x and target y 
-- The division by n can be avoided if one sets the internal variable sizeAverage to false
----------------------------------------------------------------------
function defineAbsCriterion(model, average)
  criterion = nn.AbsCriterion();
  criterion.sizeAverage = average or true;
  return model, criterion;
end

----------------------------------------------------------------------
-- Smoothed L1 Criterion
-- Can be thought of as a smooth version of the AbsCriterion 
-- It is less sensitive to outliers than the MSECriterion and in some cases prevents exploding gradients
----------------------------------------------------------------------
function defineSmoothL1Criterion(model, average)
  criterion = nn.SmoothL1Criterion();
  criterion.sizeAverage = average or true;
  return model, criterion;
end

----------------------------------------------------------------------
-- Kullback-Leibler Criterion
-- The Kullback–Leibler divergence criterion. KL divergence is a useful distance measure for continuous distributions 
-- Often useful when performing direct regression over the space
----------------------------------------------------------------------
function defineKLCriterion(model)
  criterion = nn.DistKLDivCriterion();
  return model, criterion;
end

----------------------------------------------------------------------
-- Mean Squared Error Criterion
-- Measures the mean squared error between n elements in the input x and output y 
-- The division by n can be avoided if one sets the internal variable sizeAverage to false
----------------------------------------------------------------------
function defineMSECriterion(model, average)
  criterion = nn.MSECriterion();
  criterion.sizeAverage = average or true;
  return model, criterion;
end

----------------------------------------------------------------------
-- Binary Cross Entropy
-- Criterion that measures the Binary Cross Entropy between the target and the output
-- This is used for measuring the error of a reconstruction in for example an auto-encoder.
----------------------------------------------------------------------
function defineBCECriterion(model, weights)
  if (weights) then
    criterion = nn.BCECriterion(weights);
  else
    criterion = nn.BCECriterion();
  end
  return model, criterion;
end

----------------------------------------------------------------------
-- Class Negative Log Likelihood Criterion
-- Negative log likelihood criterion. It is useful to train a classication problem with n classes. 
-- If provided, the optional argument weights should be a 1D Tensor assigning weight to each of the classes
----------------------------------------------------------------------
function defineNLLCriterion(model, weights)
  model:add(nn.LogSoftMax());
  if (weights) then
    criterion = nn.ClassNLLCriterion(weights);
  else
    criterion = nn.ClassNLLCriterion();
  end
  return model, criterion;
end

----------------------------------------------------------------------
-- Cross Entropy Criterion
-- This criterion combines LogSoftMax and ClassNLLCriterion in one single class.
-- The input given through a forward() is expected to contain scores for each class.
----------------------------------------------------------------------
function defineCrossEntropyCriterion(model, weights)
  if (weights) then
    criterion = nn.CrossEntropyCriterion(weights);
  else
    criterion = nn.CrossEntropyCriterion();
  end
  return model, criterion;
end

----------------------------------------------------------------------
-- Margin criterion
-- Optimizes a two-class classification hinge loss (margin-based loss) between input x and output y {1,-1}. 
-- Margin, if unspecified, is by default 1
----------------------------------------------------------------------
function defineMarginCriterion(model, margin, average)
  if (margin) then
    criterion = nn.MarginCriterion(margin);
  else
    criterion = nn.MarginCriterion();
  end
  criterion.sizeAverage = average or true;
  return model, criterion;
end

----------------------------------------------------------------------
-- Multi-Margin criterion
-- Optimizes a multi-class classification hinge loss (margin-based loss) between input x and output y (target class index). 
-- Especially useful for a network ending with a layer computing pairwise similarity
----------------------------------------------------------------------
function defineMultiMarginCriterion(model, p)
  model:add(nn.Euclidean(n, m));
  model:add(nn.MulConstant(-1));
  criterion = nn.MultiMarginCriterion(p);
  return model, criterion;
end

----------------------------------------------------------------------
-- Multi-Label Margin criterion
-- Optimizes a multi-class classification hinge loss (margin-based loss) between input x and output y (vector of class indexes). 
-- Especially useful for a network ending with a layer computing pairwise similarity
----------------------------------------------------------------------
function defineMultiLabelMarginCriterion(model)
  criterion = nn.MultiLabelMarginCriterion();
  return model, criterion;
end

----------------------------------------------------------------------
-- Hinge Embedding Criterion
-- Creates a criterion that measures the loss given an input x which is a 1-dimensional vector and a label y (1 or -1).
-- This is usually used for measuring whether two inputs are similar or dissimilar (using L1 distance)
-- Typically used for learning nonlinear embeddings or semi-supervised learning.
----------------------------------------------------------------------
function defineHingeEmbeddingCriterion(model, margin)
  criterion = nn.HingeEmbeddingCriterion(margin or 1);
  return model, criterion;
end

----------------------------------------------------------------------
-- L1 Hinge Embedding Criterion
-- Creates a criterion that measures the loss given an input x = {x1, x2}, a table of two Tensors, and a label y (1 or -1)
-- This is usually used for measuring whether two inputs are similar or dissimilar (using L1 distance)
-- Typically used for learning nonlinear embeddings or semi-supervised learning.
----------------------------------------------------------------------
function defineL1HingeEmbeddingCriterion(model, margin)
  criterion = nn.L1HingeEmbeddingCriterion(margin or 1);
  return model, criterion;
end

----------------------------------------------------------------------
-- Cosine Embedding Criterion
-- Creates a criterion that measures the loss given an input x = {x1, x2}, a table of two Tensors, and a label y (1 or -1)
-- This is usually used for measuring whether two inputs are similar or dissimilar (using cosine distance)
-- Typically used for learning nonlinear embeddings or semi-supervised learning.
----------------------------------------------------------------------
function defineCosineEmbeddingCriterion(model, margin)
  criterion = nn.CosineEmbeddingCriterion(margin or 1);
  return model, criterion;
end

----------------------------------------------------------------------
-- Margin Ranking Criterion
-- Creates a criterion that measures the loss given an input x = {x1, x2}, a table of two Tensors, and a label y (1 or -1)
-- If y == 1 then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for y == -1
----------------------------------------------------------------------
function defineMarginRankingCriterion(model, margin)
  criterion = nn.MarginRankingCriterion(margin or 1);
  return model, criterion;
end

----------------------------------------------------------------------
-- Multiple criterion
-- Criterion which is a weighted sum of other Criterion. 
----------------------------------------------------------------------
function defineMultiCriterion(model, criterions, weights)
  criterion = nn.MultiCriterion();
  for k, v in ipairs(criterions) do
    criterion:add(k, weights[k] or 1);
  end
  return model, criterion;
end

----------------------------------------------------------------------
-- Parallel criterion
-- Criterion which is a weighted sum of other Criterion. 
-- Oppositely to multiple, target can be presented with different inputs
----------------------------------------------------------------------
function defineParallelCriterion(model, criterions, weights, repeatTarget)
  criterion = nn.ParallelCriterion(repeatTarget or false);
  for k, v in ipairs(criterions) do
    criterion:add(k, weights[k] or 1);
  end
  return model, criterion;
end