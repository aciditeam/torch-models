-- graphical model lib
require 'gm'
local nninit = require 'nninit'

local modelMRF, parent = torch.class('modelMRF', 'modelClass')
  
function modelMRF:defineModel(structure, options)
end

-- shortcuts
local tensor = torch.Tensor
local zeros = torch.zeros
local ones = torch.ones
local randn = torch.randn
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local exp = torch.exp
local floor = torch.floor
local ceil = math.ceil
local uniform = torch.uniform

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- Example of how to train a CRF for a simple segmentation task
--
do
   -- define graph:
   nNodes = 10
   nStates = 2
   adjacency = torch.zeros(nNodes,nNodes)
   for i = 1,nNodes-1 do
      adjacency[i][i+1] = 1
      adjacency[i+1][i] = 1
   end
   g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=10, type='mrf', verbose=true}

   -- define training set:
   nInstances = 100
   Y = tensor(nInstances,nNodes)
   for i = 1,nInstances do
      -- each entry is either 1 or 2, with a probability that
      -- increases with the node index
      for n = 1,nNodes do
         Y[i][n] = torch.bernoulli((n-1)/(nNodes-1)) + 1
      end
      -- create correlation between last two nodes
      Y[i][nNodes-1] = Y[i][nNodes]
   end

   -- NOTE: the 10 training nodes in Y have probability 0, 1/9, ... , 9/9 to be equal
   -- to 2. The node beliefs obtained after training should show that.

   -- tie node potentials to parameter vector
   -- NOTE: we allocate one parameter per node, to properly model
   -- the probability of each node
   nodeMap = zeros(nNodes,nStates)
   for n = 1,nNodes do
      nodeMap[{ n,1 }] = n
   end

   -- tie edge potentials to parameter vector
   -- NOTE: we allocate parameters globally, i.e. parameters model
   -- pairwise relations globally
   nEdges = g.edgeEnds:size(1)
   edgeMap = zeros(nEdges,nStates,nStates)
   edgeMap[{ {},1,1 }] = nNodes+1
   edgeMap[{ {},2,2 }] = nNodes+2
   edgeMap[{ {},1,2 }] = nNodes+3

   -- initialize parameters
   g:initParameters(nodeMap,edgeMap)
   
   -- estimate nll:
   require 'optim'
   optim.lbfgs(function()
      local f,grad = g:nll('exact',Y)
      print('LBFGS – objective = ', f)
      return f,grad
   end, g.w, {maxIter=100, lineSearch=optim.lswolfe})

   -- gen final potentials
   g:makePotentials()

   -- exact decoding:
   local exact = g:decode('exact')
   print()
   print('<gm.testme> exact optimal config:')
   print(exact)

   -- exact inference:
   local nodeBel,edgeBel,logZ = g:infer('exact')
   print('<gm.testme> node beliefs (prob that node=2)')
   print(nodeBel[{ {},2 }])
   print('<gm.testme> edge beliefs (prob that node1=2 & node2=2)')
   print(edgeBel[{ {},2,2 }])

   -- sample from model:
   local samples = g:sample('exact',5)
   print('<gm.testme> 5 samples from model:')
   print(samples)

   local samples = g:sample('gibbs',5)
   print('<gm.testme> 5 samples from model (Gibbs):')
   print(samples)
end

function modelMRF:definePretraining(structure, l, options)
  -- TODO
  return model;
end

function modelMRF:retrieveEncodingLayer(model) 
  -- Here simply return the encoder
  encoder = model.encoder
  encoder:remove();
  return model.encoder;
end

function modelMRF:weightsInitialize(model)
  -- TODO
  return model;
end

function modelMRF:weightsTransfer(model, trainedLayers)
  -- TODO
  return model;
end

function modelMRF:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.ReLU;
  self.batchNormalize = true;
  self.pretrainType = 'ae';
  self.pretrain = true;
  self.dropout = 0.5;
end

function modelMRF:parametersRandom()
  -- All possible non-linearities
  self.distributions = {};
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
  self.distributions.batchNormalize = {true, false};
  self.distributions.pretrainType = {'ae', 'psd'};
  self.distributions.pretrain = {true, false};
  self.distributions.dropout = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
end