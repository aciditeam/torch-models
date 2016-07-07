----------------------------------------------------------------------
--
-- Deep time series learning: Parameter optimization
--
-- Main script for network topology
-- Philippe Esling
-- <esling@ircam.fr>
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
----------------------------------------------------------------------
require 'torch'
require 'importTSDataset'

----------------------------------------------------------------------
-- Module variable
----------------------------------------------------------------------
local hyperParameters = torch.class("hyperParameters")

function hyperParameters:__init(sampler)
  self.parameters = {};   -- Name of optimized parameters
  self.type = {};         -- Type (for sampling)
  self.range = {};        -- Range of values (bounds or categories)
  self.default = {};      -- Default values
  self.past = {};         -- Past parameters values
  self.pastMatrix = {};   -- Past values matrix
  self.ranks = {};        -- Ranks of configurations
  self.active = {};       -- Active state of parameters
  self.errors = {};       -- Complete error matrix
  self.errorMeans = {};   -- Mean of errors
  self.rangeSteps = {};   -- Steps of evaluation (for fit)
  self.isFitted = {};
  self.nextBatch = {};    -- Set of fitted parameters
  self.nbParameters = 0;  -- Number of optimized parameters
  self.nbEvaluated = 0;   -- Number of configurations evaluated
  self.nbNetworks = 0;    -- Number of planned configurations
  self.nbDatasets = 0;    -- Number of datasets
  self.nbRepeat = 0;      -- Repetitions per configuration
  self.currentBatch = 0;  -- Current used set of parameters
  self.curEval = 0;
  -- Sampler to use 
  self.sampler = sampler;
end

----------------------------------------------------------------------
-- Register a particular parameter
----------------------------------------------------------------------
function hyperParameters:registerParameter(name, type, values)
  self.nbParameters = self.nbParameters + 1;
  self.parameters[name] = name;
  self.type[name] = type;
  self.range[name] = values;
end

----------------------------------------------------------------------
-- Unregister all parameters
----------------------------------------------------------------------
function hyperParameters:unregisterAll()
  self.parameters = {};
  self.type = {};
  self.range = {};
  self.default = {};
  self.past = {};
  self.pastMatrix = {};
  self.ranks = {};
  self.active = {};
  self.errors = {};
  self.errorMeans = {};
  self.rangeSteps = {};
  self.nextBatch = {};
  self.isFitted = {};
  self.nbParameters = 0;
  self.nbEvaluated = 0;
  self.nbNetworks = 0;
  self.nbDatasets = 0;
  self.nbRepeat = 0;
  self.curEval = 0;
  self.currentBatch = 0;
  collectgarbage();
end

----------------------------------------------------------------------
-- Register a particular parameter
----------------------------------------------------------------------
function hyperParameters:initStructure(nbNetworks, nbDatasets, nbRepeat)
  -- Ranges of step evaluation for kernel fitting
  self.rangeSteps = torch.Tensor(self.nbParameters);
  self.nextBatch = torch.Tensor(100, self.nbParameters);
  curParam = 1;
  for k,v in pairs(self.parameters) do
    self.parameters[k] = {};
    -- Values of parameters
    self.past[k] = torch.Tensor(nbNetworks);
    -- Which were active
    self.active[k] = torch.ones(nbNetworks);
    -- Step for grid fitting
    if      (self.type[k] == 'real' or self.type[k] == 'int')       then self.rangeSteps[curParam] = ((self.range[k][2] - self.range[k][1]) / 100);
    elseif  (self.type[k] == 'catInt' or self.type[k] == 'catStr')  then self.rangeSteps[curParam] = 1;
    end
    for n = 1, 100 do self.nextBatch[n][curParam] = self.range[k][1]; end
    curParam = curParam + 1;
  end
  -- Corresponding mean ranks (over full)
  self.ranks = torch.Tensor(nbNetworks);
  -- Independent error values over repetitions
  self.errors = torch.Tensor(nbNetworks, nbDatasets, nbRepeat);
  -- Mean error values over repetitions
  self.errorMeans = torch.Tensor(nbNetworks, nbDatasets);
  -- Matrix of past values
  self.pastMatrix = torch.Tensor(nbNetworks, self.nbParameters);
  -- Keep track of which networks have been fitted
  self.isFitted = torch.Tensor(nbNetworks);
  -- Keep evaluation values
  self.nbNetworks = nbNetworks;
  self.nbDatasets = nbDatasets;
  self.nbRepeat = nbRepeat;
end

----------------------------------------------------------------------
-- Output hyperparameters of the network to a file
----------------------------------------------------------------------
function hyperParameters:outputResults(fID)
  -- Simplification function
  printf = function(f, s, ...) return f:write(s:format(...)) end -- function
  -- output parameter names
  for k,v in pairs (self.parameters) do
    printf(fID, '%s\t', k);
  end
  printf(fID, 'Fitted\t');
  printf(fID, 'Mean rank\t');
  for i = 1,self.nbDatasets do
    printf(fID, 'Mean e.%d\t', i);
  end
  printf(fID, '\n');
  -- parse through various evaluations
  for n = 1,self.nbEvaluated do
    for k,v in pairs (self.parameters) do
      -- output parameter value
      if      self.type[k] == 'real'    then printf(fID, '%f\t', self.past[k][n]);
      elseif  self.type[k] == 'int'     then printf(fID, '%d\t', self.past[k][n]);
      elseif  self.type[k] == 'catInt'  then printf(fID, '%d\t', self.range[k][self.past[k][n]]);
      elseif  self.type[k] == 'catStr'  then printf(fID, '%s\t', self.range[k][self.past[k][n]]);
      end
    end
    printf(fID, '%d\t', self.isFitted[n]);
    printf(fID, '%f\t', self.ranks[n]);
    for d = 1,self.nbDatasets do
      printf(fID, '%f\t', self.errorMeans[n][d]);
    end
    printf(fID, '\n')
  end
end

----------------------------------------------------------------------
-- Create a past values structure to feed to a kernel
----------------------------------------------------------------------
function hyperParameters:currentPast()
  -- Temporary past matrix
  local currentPast = torch.Tensor(self.nbEvaluated, self.nbParameters);
  local curParam = 1;
  -- Parse through the registered parameters
  for k,v in pairs (self.parameters) do
    currentPast[{{}, curParam}] = self.past[k][{{1, self.nbEvaluated}}];
    curParam = curParam + 1;
  end
  return currentPast, self.ranks[{{1, self.nbEvaluated}}];
end

----------------------------------------------------------------------
-- Output comparison between train and untrained
----------------------------------------------------------------------
function hyperParameters:outputComparison(fID, n, errors)
  -- Simplification function
  printf = function(f, s,...) return f:write(s:format(...)) end -- function
  -- parse through various evaluations
  for k,v in pairs (self.parameters) do
    -- output parameter value
    if      self.type[k] == 'real'    then printf(fID, '%f\t', self.past[k][n]);
    elseif  self.type[k] == 'int'     then printf(fID, '%d\t', self.past[k][n]);
    elseif  self.type[k] == 'catInt'  then printf(fID, '%d\t', self.range[k][self.past[k][n]]);
    elseif  self.type[k] == 'catStr'  then printf(fID, '%s\t', self.range[k][self.past[k][n]]);
    end
  end
  for d = 1,self.nbDatasets do
    printf(fID, '%f\t', self.errorMeans[n][d]);
  end
  for d = 1,self.nbDatasets do
    printf(fID, '%f\t', errors[d]);
  end
  printf(fID, '\n')
end

----------------------------------------------------------------------
-- Sample from the past values (depending on previous ranks)
----------------------------------------------------------------------
function hyperParameters:samplePast(nNet, nbTrainNetworks)
  -- Obtain the current ranks
  local curRanks = self.ranks[{{1, self.nbEvaluated}}];
  -- Sort them in ascending order
  vals, bestIDs = torch.sort(curRanks);
  -- Take the ID of the current "slice"
  local curSlice = torch.round((nNet - 1) * (self.nbEvaluated / nbTrainNetworks)) + 1;
  -- "Fake fill" the next parameters
  for k,v in pairs(self.parameters) do
      self.past[k][self.nbEvaluated+1] = self.past[k][bestIDs[curSlice]]; 
  end
  -- Keep the current ID evaluated
  self.curEval = bestIDs[curSlice];
  return self.curEval;
end

----------------------------------------------------------------------
-- Output hyperparameters of the network to a file
----------------------------------------------------------------------
function hyperParameters:randomDraw()
  -- Depending on the current
  local proba = self.sampler:uniform(0, 1);
  -- At the beginning favor the random networks
  if proba > (self.nbEvaluated / self.nbNetworks) then
    self.isFitted[self.nbEvaluated+1] = 0;
    local curParam = 1;
    -- Parse through the registered parameters
    for k,v in pairs(self.parameters) do
      if      self.type[k] == 'real'    then self.past[k][self.nbEvaluated+1] = self.sampler:uniform(self.range[k][1],self.range[k][2]);
      elseif  self.type[k] == 'int'     then self.past[k][self.nbEvaluated+1] = self.sampler:randint(self.range[k][1],self.range[k][2]);
      elseif  self.type[k] == 'catInt'  then self.past[k][self.nbEvaluated+1] = self.sampler:randint(1,#self.range[k]);
      elseif  self.type[k] == 'catStr'  then self.past[k][self.nbEvaluated+1] = self.sampler:randint(1,#self.range[k]);
      end
      self.pastMatrix[self.nbEvaluated+1][curParam] = self.past[k][self.nbEvaluated+1];
      curParam = curParam + 1;
    end
  else
    self.isFitted[self.nbEvaluated+1] = 1;
    -- Here we extract one from the pre-computed next batch
    self.currentBatch = self.currentBatch + 1;
    local curParam = 1;
    -- Draw from the pre-computed batch
    for k,v in pairs(self.parameters) do
      self.past[k][self.nbEvaluated+1] = self.nextBatch[self.currentBatch][curParam];
      self.pastMatrix[self.nbEvaluated+1][curParam] = self.past[k][self.nbEvaluated+1];
      curParam = curParam + 1;
    end
  end
end

----------------------------------------------------------------------
-- Retrieve current value of a parameter
----------------------------------------------------------------------
function hyperParameters:getCurrentParameter(k)
  local val
  if      self.type[k] == 'real'    then val = self.past[k][self.nbEvaluated+1];
  elseif  self.type[k] == 'int'     then val = self.past[k][self.nbEvaluated+1];
  elseif  self.type[k] == 'catInt'  then val = self.range[k][self.past[k][self.nbEvaluated+1]];
  elseif  self.type[k] == 'catStr'  then val = self.range[k][self.past[k][self.nbEvaluated+1]];
  end
  return val
end

----------------------------------------------------------------------
-- Generate a given number of random networks
----------------------------------------------------------------------
function hyperParameters:generateRandomBatch(nbRandom)
  local randomBatch = torch.Tensor(nbRandom, self.nbParameters);
  local val = 0;
  for n = 1,nbRandom do
    local curParam = 1;
    for k,v in pairs(self.parameters) do
      if      self.type[k] == 'real'    then val = self.sampler:uniform(self.range[k][1],self.range[k][2]);
      elseif  self.type[k] == 'int'     then val = self.sampler:randint(self.range[k][1],self.range[k][2]);
      elseif  self.type[k] == 'catInt'  then val = self.sampler:randint(1,#self.range[k]);
      elseif  self.type[k] == 'catStr'  then val = self.sampler:randint(1,#self.range[k]);
      end
      randomBatch[n][curParam] = val;
      curParam = curParam + 1;
    end
  end
  return randomBatch;
end

----------------------------------------------------------------------
-- Register the current error values
----------------------------------------------------------------------
function hyperParameters:registerResults(errors)
  -- Increase number of evaluated
  self.nbEvaluated = self.nbEvaluated + 1;
  -- Independent error values over repetitions
  self.errors[{self.nbEvaluated, {}, {}}] = errors;
  -- Mean error values over repetitions
  self.errorMeans[{self.nbEvaluated, {}}] = errors:mean(2);
end

----------------------------------------------------------------------
-- Update the current values of the mean ranks
----------------------------------------------------------------------
function hyperParameters:updateRanks()
  -- Simply recompute the critical differences between mean errors
  self.ranks[{{1,self.nbEvaluated}}] = self:criticalDifference(self.errorMeans[{{1,self.nbEvaluated}, {}}]);
end

----------------------------------------------------------------------
-- Use a kernel to predict (fit) the error and variance of randomly distributed hyper-parameters
----------------------------------------------------------------------
function hyperParameters:fit(nbRandom, nbBatch)
  print('- Generating next random batch.');
  -- For now generate a given number of networks
  local finalGrid = self:generateRandomBatch(nbRandom);
  -- Retrieve values and ranks
  local curValues, curRanks
  curValues, curRanks = self:currentPast();
  -- We estimate the best infered values from Nadaraya-Watson kernel regression
  print('- Fitting Nadaraya-Watson kernel.');
  kernel = self:ksrmv(curValues, curRanks, self.rangeSteps, finalGrid);
  -- Prediction of what would be the best hyperparameters
  _, bestIDs = torch.sort(kernel.f);
  -- Prepare the next batch
  print('- Selecting best hyper-parameters.');
  self.nextBatch = finalGrid[{{bestIDs[{{1,nbBatch}}]}, {}}];
end

----------------------------------------------
-- Produces the critical difference of the statistical significance of a matrix of
-- scores, S, achieved by a set of machine learning algorithms.
--
-- References
-- [1] Demsar, J., "Statistical comparisons of classifiers over multiple
--     datasets", Journal of Machine Learning Research, vol. 7, pp. 1-30,
--     2006.
--
function hyperParameters:criticalDifference(s)  
  -- Convert scores into ranks
  N = s:size(1);
  k = s:size(2);
  S,r = torch.sort(s:t());
  S = S:t();
  -- Retrieve unique values in S
  idx = torch.range(1,N);
  ranks = torch.zeros(N, k);
  for j = 1,k do
    local curVals = {};
    local curRanks = torch.zeros(N);
    for i = 1,N do
      if curVals[S[i][j]] == nil then
        curVals[S[i][j]] = true;
        selected = idx[s[{{}, j}]:eq(S[i][j])];
        for t = 1,selected:size(1) do
          curRanks[selected[t]] = i;
        end
      end
    end
    ranks[{{}, j}] = curRanks;
  end
  ranks = torch.mean(ranks, 2);
  return ranks;
end

----------------------------------------------------
-- KSRMV   Multivariate kernel smoothing regression
--
-- r=ksrmv(x,y) returns the Gaussian kernel regression in structure r such that
--   r.f(r.x) = y(x) + e
-- f(x) = sum(kerf((x-X)/h).*Y)/sum(kerf((x-X)/h))
--
function hyperParameters:ksrmv(x,y,hx,z)
  -- Start by reshaping input
  if x:size(1) ~= y:size(1) then
    error('x and y have different rows.');
  end
  d = x:size(2);
  -- Default parameters
  if z:size(2) ~= d then
    error('z must have the same number of columns as x.')
  end
  local r = {};
  r.x = z;
  N = z:size(1);
  r.n = y:size(1);
  if y == {} then
    -- Optimal bandwidth suggested by Bowman and Azzalini (1997) p.31
    hy = torch.median(torch.abs(y - torch.median(y))) / 0.6745 * (4 / (d + 2) / r.n) ^ (1 / (d + 4));
    hx = torch.median(abs(x - torch.median(x):repeatTensor(r.n, 1))) / 0.6745 * (4 / (d + 2) / r.n) ^ (1 / (d + 4));
    hx = sqrt(hy * hx);
  elseif hx:size(1) ~= d then
    error('h must be a scalar.')
  end
  r.h = hx;
  -- Improved efficient code
  -- Scaling first
  H = torch.diag(torch.ones(hx:size(1)):cdiv(hx));
  x = x * H;
  x1 = r.x * H;
  -- Gaussian kernel function
  kerf = function(z) return torch.exp(-torch.sum(torch.cmul(z,z), 2) / 2); end
  -- Allocate memory
  r.f = torch.zeros(N,1);
  -- Loop through each regression point
  for k = 1,N do
    -- scaled deference from regression point
    xx = torch.abs(x - x1[{k,{}}]:repeatTensor(r.n));
    -- select neighbours using exp(-5^2/2)<5e-6
    idx = xx:lt(5)
    -- kernel function
    z = kerf(xx[{{idx},{}}]);
    -- regression
    r.f[k] = torch.sum(torch.cmul(z,y[{{idx}}])) / torch.sum(z);
  end
  return r;
end
