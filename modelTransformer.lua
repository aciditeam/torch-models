----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Main functions for transformer network
--
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'image'
local nninit = require 'nninit'

local BilinearSamplerTS, parent = torch.class('nn.BilinearSamplerTS', 'nn.Module')

----------------------------------------------------------------------

--[[
   BilinearSamplerTS will perform bilinear sampling of the input series according to the
   normalized coordinates provided in the grid. Output will be of same size as the grids, 
   with as many features as the input series.
   - inputSeries has to be in BHWD layout
   - grids have to be in BHWD layout, with dim(D)=2
   - grids contains, for each sample (first dim), the normalized coordinates of the output wrt the input sample
      - first coordinate is Y coordinate, second is X
      - normalized coordinates : (-1,-1) points to top left, (-1,1) points to top right
      - if the normalized coordinates fall outside of the image, then output will be filled with zeros
]]

----------------------------------------------------------------------

function BilinearSamplerTS:__init()
  parent.__init(self)
  self.gradInput={}
end

function BilinearSamplerTS:check(input, gradOutput)
  local inputSeries = input[1]
  local grids = input[2]
  assert(inputSeries:nDimension() == 3)
  assert(grids:nDimension() == 2)
  assert(inputSeries:size(1) == grids:size(1)) -- batch
  assert(grids:size(4)==2) -- coordinates

  if gradOutput then
    assert(grids:size(1)==gradOutput:size(1))
    assert(grids:size(2)==gradOutput:size(2))
    assert(grids:size(3)==gradOutput:size(3))
  end
end

local function addOuterDim(t)
  local sizes = t:size()
  local newsizes = torch.LongStorage(sizes:size()+1)
  newsizes[1]=1
  for i=1,sizes:size() do
    newsizes[i+1]=sizes[i]
  end
  return t:view(newsizes)
end

function BilinearSamplerTS:updateOutput(input)
  local _inputSeries = input[1]
  local _grids = input[2]

  local inputSeries, grids
  if _inputSeries:nDimension() == 3 then
    inputSeries = addOuterDim(_inputSeries)
    grids = addOuterDim(_grids)
  else
    inputSeries = _inputSeries
    grids = _grids
  end

  local input = {inputSeries, grids}
  self:check(input)
  self.output:resize(inputSeries:size(1), grids:size(2), grids:size(3), inputSeries:size(4))
  inputSeries.nn.BilinearSamplerTS_updateOutput(self, inputSeries, grids)
  if _inputSeries:nDimension() == 3 then
    self.output=self.output:select(1,1)
  end
  return self.output
end

function BilinearSamplerTS:updateGradInput(_input, _gradOutput)
  local _inputImages = _input[1]
  local _grids = _input[2]

  local inputImages, grids, gradOutput
  if _inputImages:nDimension() == 3 then
    inputImages = addOuterDim(_inputImages)
    grids = addOuterDim(_grids)
    gradOutput = addOuterDim(_gradOutput)
  else
    inputImages = _inputImages
    grids = _grids
    gradOutput = _gradOutput
  end

  local input = {inputImages, grids}

  self:check(input, gradOutput)
  for i=1,#input do
    self.gradInput[i] = self.gradInput[i] or input[1].new()
    self.gradInput[i]:resizeAs(input[i]):zero()
  end

  local gradInputImages = self.gradInput[1]
  local gradGrids = self.gradInput[2]

  inputImages.nn.BilinearSamplerTS_updateGradInput(self, inputImages, grids, gradInputImages, gradGrids, gradOutput)

  if _gradOutput:nDimension()==3 then
    self.gradInput[1]=self.gradInput[1]:select(1,1)
    self.gradInput[2]=self.gradInput[2]:select(1,1)
  end

  return self.gradInput
end

local ATMG, parent = torch.class('nn.AffineTransformMatrixGenerator', 'nn.Module')

----------------------------------------------------------------------

--[[
AffineTransformMatrixGenerator(useRotation, useScale, useTranslation) :
AffineTransformMatrixGenerator:updateOutput(transformParams)
AffineTransformMatrixGenerator:updateGradInput(transformParams, gradParams)
This module can be used in between the localisation network (that outputs the
parameters of the transformation) and the AffineGridGeneratorBHWD (that expects
an affine transform matrix as input).
The goal is to be able to use only specific transformations or a combination of them.
If no specific transformation is specified, it uses a fully parametrized
linear transformation and thus expects 6 parameters as input. In this case
the module is equivalent to nn.View(2,3):setNumInputDims(2).
Any combination of the 3 transformations (rotation, scale and/or translation)
can be used. The transform parameters must be supplied in the following order:
rotation (1 param), scale (1 param) then translation (2 params).
Example:
AffineTransformMatrixGenerator(true,false,true) expects as input a tensor of
if size (B, 3) containing (rotationAngle, translationX, translationY).
]]

----------------------------------------------------------------------

function ATMG:__init(useRotation, useScale, useTranslation)
  parent.__init(self)

  -- if no specific transformation, use fully parametrized version
  self.fullMode = not(useRotation or useScale or useTranslation)

  if not self.fullMode then
    self.useRotation = useRotation
    self.useScale = useScale
    self.useTranslation = useTranslation
  end
end

function ATMG:check(input)
  if self.fullMode then
    assert(input:size(2)==6, 'Expected 6 parameters, got ' .. input:size(2))
  else
    local numberParameters = 0
    if self.useRotation then
      numberParameters = numberParameters + 1
    end
    if self.useScale then
      numberParameters = numberParameters + 1
    end
    if self.useTranslation then
      numberParameters = numberParameters + 2
    end
    assert(input:size(2)==numberParameters, 'Expected '..numberParameters..
      ' parameters, got ' .. input:size(2))
  end
end

local function addOuterDim(t)
  local sizes = t:size()
  local newsizes = torch.LongStorage(sizes:size()+1)
  newsizes[1]=1
  for i=1,sizes:size() do
    newsizes[i+1]=sizes[i]
  end
  return t:view(newsizes)
end

function ATMG:updateOutput(_tranformParams)
  local transformParams
  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
  else
    transformParams = _tranformParams
  end

  self:check(transformParams)
  local batchSize = transformParams:size(1)

  if self.fullMode then
    self.output = transformParams:view(batchSize, 2, 3)
  else
    local completeTransformation = torch.zeros(batchSize,3,3):typeAs(transformParams)
    completeTransformation:select(3,1):select(2,1):add(1)
    completeTransformation:select(3,2):select(2,2):add(1)
    completeTransformation:select(3,3):select(2,3):add(1)
    local transformationBuffer = torch.Tensor(batchSize,3,3):typeAs(transformParams)

    local paramIndex = 1
    if self.useRotation then
      local alphas = transformParams:select(2, paramIndex)
      paramIndex = paramIndex + 1

      transformationBuffer:zero()
      transformationBuffer:select(3,3):select(2,3):add(1)
      local cosines = torch.cos(alphas)
      local sinuses = torch.sin(alphas)
      transformationBuffer:select(3,1):select(2,1):copy(cosines)
      transformationBuffer:select(3,2):select(2,2):copy(cosines)
      transformationBuffer:select(3,1):select(2,2):copy(sinuses)
      transformationBuffer:select(3,2):select(2,1):copy(-sinuses)

      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    end
    self.rotationOutput = completeTransformation:narrow(2,1,2):narrow(3,1,2):clone()

    if self.useScale then
      local scaleFactors = transformParams:select(2,paramIndex)
      paramIndex = paramIndex + 1

      transformationBuffer:zero()
      transformationBuffer:select(3,1):select(2,1):copy(scaleFactors)
      transformationBuffer:select(3,2):select(2,2):copy(scaleFactors)
      transformationBuffer:select(3,3):select(2,3):add(1)

      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    end
    self.scaleOutput = completeTransformation:narrow(2,1,2):narrow(3,1,2):clone()

    if self.useTranslation then
      local txs = transformParams:select(2,paramIndex)
      local tys = transformParams:select(2,paramIndex+1)

      transformationBuffer:zero()
      transformationBuffer:select(3,1):select(2,1):add(1)
      transformationBuffer:select(3,2):select(2,2):add(1)
      transformationBuffer:select(3,3):select(2,3):add(1)
      transformationBuffer:select(3,3):select(2,1):copy(txs)
      transformationBuffer:select(3,3):select(2,2):copy(tys)

      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    end

    self.output=completeTransformation:narrow(2,1,2)
  end

  if _tranformParams:nDimension()==1 then
    self.output = self.output:select(1,1)
  end
  return self.output
end


function ATMG:updateGradInput(_tranformParams, _gradParams)
  local transformParams, gradParams
  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
    gradParams = addOuterDim(_gradParams):clone()
  else
    transformParams = _tranformParams
    gradParams = _gradParams:clone()
  end

  local batchSize = transformParams:size(1)
  if self.fullMode then
    self.gradInput = gradParams:view(batchSize, 6)
  else
    local paramIndex = transformParams:size(2)
    self.gradInput:resizeAs(transformParams)
    if self.useTranslation then
      local gradInputTranslationParams = self.gradInput:narrow(2,paramIndex-1,2)
      local tParams = torch.Tensor(batchSize, 1, 2):typeAs(transformParams)
      tParams:select(3,1):copy(transformParams:select(2,paramIndex-1))
      tParams:select(3,2):copy(transformParams:select(2,paramIndex))
      paramIndex = paramIndex-2

      local selectedOutput = self.scaleOutput
      local selectedGradParams = gradParams:narrow(2,1,2):narrow(3,3,1):transpose(2,3)
      gradInputTranslationParams:copy(torch.bmm(selectedGradParams, selectedOutput))

      local gradientCorrection = torch.bmm(selectedGradParams:transpose(2,3), tParams)
      gradParams:narrow(3,1,2):narrow(2,1,2):add(1,gradientCorrection)
    end

    if self.useScale then
      local gradInputScaleparams = self.gradInput:narrow(2,paramIndex,1)
      local sParams = transformParams:select(2,paramIndex)
      paramIndex = paramIndex-1

      local selectedOutput = self.rotationOutput
      local selectedGradParams = gradParams:narrow(2,1,2):narrow(3,1,2)
      gradInputScaleparams:copy(torch.cmul(selectedOutput, selectedGradParams):sum(2):sum(3))

      gradParams:select(3,1):select(2,1):cmul(sParams)
      gradParams:select(3,2):select(2,1):cmul(sParams)
      gradParams:select(3,1):select(2,2):cmul(sParams)
      gradParams:select(3,2):select(2,2):cmul(sParams)
    end

    if self.useRotation then
      local gradInputRotationParams = self.gradInput:narrow(2,paramIndex,1)
      local rParams = transformParams:select(2,paramIndex)

      local rotationDerivative = torch.zeros(batchSize, 2, 2):typeAs(rParams)
      torch.sin(rotationDerivative:select(3,1):select(2,1),-rParams)
      torch.sin(rotationDerivative:select(3,2):select(2,2),-rParams)
      torch.cos(rotationDerivative:select(3,1):select(2,2),rParams)
      torch.cos(rotationDerivative:select(3,2):select(2,1),rParams):mul(-1)
      local selectedGradParams = gradParams:narrow(2,1,2):narrow(3,1,2)
      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(2):sum(3))
    end
  end

  if _tranformParams:nDimension()==1 then
    self.gradInput = self.gradInput:select(1,1)
  end
  return self.gradInput
end

----------------------------------------------------------------------

--[[
   AffineGridGeneratorTS(height, width) :
   AffineGridGeneratorTS:updateOutput(transformMatrix)
   AffineGridGeneratorTS:updateGradInput(transformMatrix, gradGrids)
   AffineGridGeneratorTS will take 2x3 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.
   AffineGridGenerator 
   - takes (B,2,3)-shaped transform matrices as input (B=batch).
   - outputs a grid in BHWD layout, that can be used directly with BilinearSamplerBHWD
   - initialization of the previous layer should biased towards the identity transform :
      | 1  0  0 |
      | 0  1  0 |
   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]--

----------------------------------------------------------------------
local AGG, parent = torch.class('nn.AffineGridGeneratorTS', 'nn.Module')

function AGG:__init(size)
  parent.__init(self)
  -- Check the supplied size
  assert(size > 1)
  self.size = size
  -- Generate an empty grid
  self.baseGrid = torch.Tensor(size, 3)
  for i=1,self.size do
    self.baseGrid:select(2,1):select(1,i):fill(-1 + (i-1)/(self.size-1) * 2)
  end
  for j=1,self.size do
    self.baseGrid:select(2,2):select(2,j):fill(-1 + (j-1)/(self.size-1) * 2)
  end
  self.baseGrid:select(2, 3):fill(1)
  self.batchGrid = torch.Tensor(1, size, 3):copy(self.baseGrid)
end

local function addOuterDim(t)
  local sizes = t:size()
  local newsizes = torch.LongStorage(sizes:size()+1)
  newsizes[1]=1
  for i=1,sizes:size() do
    newsizes[i+1]=sizes[i]
  end
  return t:view(newsizes)
end

function AGG:updateOutput(_transformMatrix)
  local transformMatrix
  if _transformMatrix:nDimension()==2 then
    transformMatrix = addOuterDim(_transformMatrix)
  else
    transformMatrix = _transformMatrix
  end
  assert(transformMatrix:nDimension()==3
    and transformMatrix:size(2)==2
    and transformMatrix:size(3)==3
    , 'please input affine transform matrices (bx2x3)')
  local batchsize = transformMatrix:size(1)

  if self.batchGrid:size(1) ~= batchsize then
    self.batchGrid:resize(batchsize, self.height, self.width, 3)
    for i=1,batchsize do
      self.batchGrid:select(1,i):copy(self.baseGrid)
    end
  end

  self.output:resize(batchsize, self.height, self.width, 2)
  local flattenedBatchGrid = self.batchGrid:view(batchsize, self.width*self.height, 3)
  local flattenedOutput = self.output:view(batchsize, self.width*self.height, 2)
  torch.bmm(flattenedOutput, flattenedBatchGrid, transformMatrix:transpose(2,3))
  if _transformMatrix:nDimension()==2 then
    self.output = self.output:select(1,1)
  end
  return self.output
end

function AGG:updateGradInput(_transformMatrix, _gradGrid)
  local transformMatrix, gradGrid
  if _transformMatrix:nDimension()==2 then
    transformMatrix = addOuterDim(_transformMatrix)
    gradGrid = addOuterDim(_gradGrid)
  else
    transformMatrix = _transformMatrix
    gradGrid = _gradGrid
  end

  local batchsize = transformMatrix:size(1)
  local flattenedGradGrid = gradGrid:view(batchsize, self.width*self.height, 2)
  local flattenedBatchGrid = self.batchGrid:view(batchsize, self.width*self.height, 3)
  self.gradInput:resizeAs(transformMatrix):zero()
  self.gradInput:baddbmm(flattenedGradGrid:transpose(2,3), flattenedBatchGrid)
  -- torch.baddbmm doesn't work on cudatensors for some reason

  if _transformMatrix:nDimension()==2 then
    self.gradInput = self.gradInput:select(1,1)
  end

  return self.gradInput
end

local modelTransformer, parent = torch.class('modelTransformer', 'modelCNN')


  -- Get the number of output elements for a table of convolution layers
  local function convNOut(convs, inputSize)
    print(convs[1]:get(1));
    -- Get the number of channels for conv that are multiscale or not
    local nbr_input_channels = convs[1]:get(2).inputFrameSize or convs[1]:get(1):get(2).inputFrameSize
    local output = torch.Tensor(1, inputSize, 1)
    for _, conv in ipairs(convs) do
      output = conv:forward(output)
    end
    return output:nElement(), output:size(3)
  end

----------------------------------------------------------------------
-- Temporal version of Spatial Transformer network
----------------------------------------------------------------------
function modelTransformer:defineModel(structure, options)
  -- Handle the use of CUDA
  if options.cuda then local nn = require 'cunn' else local nn = require 'nn' end
  -- Define the model
  local model = nn.Sequential();
  -- Function to create a convolution module
  local function newConvolution(dSize, nIn, nOut, multiscale, noNorm, filterSize)
    multiscale = multiscale or false
    noNorm = noNorm or false
    filterSize = filterSize or 5
    local padding_size = 2
    local pooling_size = 2
    local conv = {};
    -- Convolutional layer
    local first = nn.Sequential()
    -- Padding
    first:add(nn.Padding(2, -(filterSize/2))); first:add(nn.Padding(2, filterSize/2));
    -- Temporal convolution
    first:add(nn.TemporalConvolution(nIn, nOut, filterSize, 1))
    -- Eventual normalization
    if self.batchNormalize then
      first:add(nn.Reshape(dSize * nOut)); 
      first:add(nn.BatchNormalization(dSize * nOut));
      first:add(nn.Reshape(dSize, nOut)); 
    end
    -- Non-linearity
    first:add(self.nonLinearity())
    -- Pooling
    first:add(nn.TemporalMaxPooling(pooling_size, pooling_size));
    -- Multiscale processing (residual network type)
    if multiscale then
      conv = nn.Sequential()
      second = nn.TemporalMaxPooling(pooling_size, pooling_size);
      local parallel = nn.ConcatTable();
      parallel:add(first);
      parallel:add(second);
      conv:add(parallel);
      conv:add(nn.JoinTable(1,3));
    else
      conv = first
    end
    return conv
  end
  -- Creates a fully connection layer with the specified size.
  local function newFc(nIn, nOut)
    local fc = nn.Sequential()
    fc:add(nn.View(nIn));
    fc:add(nn.Linear(nIn, nOut));
    if self.batchNormalize then fc:add(nn.BatchNormalization(nOut)) end
    fc:add(self.nonLinearity());
    return fc
  end
  -- Creates a spatial transformer module
  -- locnet are the parameters to create the localization network
  -- rot, sca, tra can be used to force specific transformations
  -- input_size is the height (=width) of the input
  -- input_channels is the number of channels in the input
  local function newTransform(locnet, rot, sca, tra, input_size, input_channels)
    local nbr_elements = {}
    for c in string.gmatch(locnet, "%d+") do
      nbr_elements[#nbr_elements + 1] = tonumber(c)
    end
    -- Get number of params and initial state
    local init_bias = {}
    local nbr_params = 0
    -- Rotation
    if rot then nbr_params = nbr_params + 1; init_bias[nbr_params] = 0; end
    -- Scaling
    if sca then nbr_params = nbr_params + 1; init_bias[nbr_params] = 1 end
    -- Translation
    if tra then nbr_params = nbr_params + 2; init_bias[nbr_params-1] = 0; init_bias[nbr_params] = 0 end
    -- Fully parametrized case
    if nbr_params == 0 then nbr_params = 6; init_bias = {1,0,0,0,1,0} end
    -- Actual spatial transformer network
    local st = nn.Sequential()
    -- Create a localization network same as cnn but with downsampled inputs
    local localization_network = nn.Sequential()
    local conv1 = newConvolution(input_size / 2, 1, nbr_elements[1], false, true, 5)
    local conv2 = newConvolution(input_size / 4, nbr_elements[1], nbr_elements[2], false, true, 5)
    local conv_output_size = input_size/4 * nbr_elements[2];
    local fc = newFc(conv_output_size, nbr_elements[3])
    local classifier = nn.Linear(nbr_elements[3], nbr_params)
    -- Initialize the localization network (see paper, A.3 section)
    classifier.weight:zero()
    classifier.bias = torch.Tensor(init_bias)
    -- Combine all the localization network together
    localization_network:add(nn.TemporalMaxPooling(2, 2))
    localization_network:add(conv1)
    localization_network:add(conv2)
    localization_network:add(fc)
    localization_network:add(classifier)
    -- Create the actual module structure
    local ct = nn.ConcatTable()
    local branch1 = nn.Sequential()
    -- The first branch just forwards the output (for future sampling)
    branch1:add(nn.Identity())
    -- Second branch compuptes the actual transform
    local branch2 = nn.Sequential()
    branch2:add(localization_network)
    -- Generate an optimal (and differentiable) transform matrix
    branch2:add(nn.AffineTransformMatrixGenerator(rot, sca, tra))
    -- Extract the grid implied by the transform
    branch2:add(nn.AffineGridGeneratorTS(input_size, input_size))
    -- Separate the two branches
    ct:add(branch1)
    ct:add(branch2)
    st:add(ct)
    -- And then sample from the original branch
    local sampler = nn.BilinearSamplerTS()
    if not no_cuda then
      sampler:type('torch.FloatTensor')
      -- make sure it will not go back to the GPU when we call :cuda()
      sampler.type = function(type)
        return self
      end
      st:add(sampler)
    else
      st:add(sampler)
    end
    --st:add(nn.Transpose({2,4},{3,4}))
    return st
  end
  -- Main construction of the network
  local network = nn.Sequential();
  network:add(nn.Reshape(structure.nInputs, 1));
  local nbr_elements = {}
  for c in string.gmatch(self.cnn, "%d+") do
    nbr_elements[#nbr_elements + 1] = tonumber(c)
  end
  assert(#nbr_elements == 3, 'opt.cnn should contain 3 comma separated values, got '..#nbr_elements)
  local conv1 = newConvolution(structure.nInputs, structure.nInputs, nbr_elements[1], false, self.no_cnorm)
  local conv2 = newConvolution(structure.nInputs, nbr_elements[1], nbr_elements[2], self.ms, self.no_cnorm)
  --local conv_output_size = convNOut({conv1, conv2}, structure.nInputs)
  local fc = newFc(nbr_elements[2] * (structure.nInputs / 4), nbr_elements[3])
  local classifier = nn.Linear(nbr_elements[3], structure.nOutputs)
  -- Add the first transformer network layer
  network:add(newTransform(self.locnet, self.rot, self.sca, self.tra, structure.nInputs, structure.nInputs, self.no_cuda))
  -- Add the convolutional layer
  network:add(conv1)
  -- Check the current output numbers
  local current_size = nbr_elements[1] * (structure.nInputs / 4);
  -- Add the second transformer layer 
  network:add(newTransform(self.locnet2, self.rot, self.sca, self.tra, current_size, nbr_elements[1], self.no_cuda))
  -- Add the second convolutional layer
  network:add(conv2)
  -- Add the fully connected network
  network:add(fc)
  -- Add the classifier
  network:add(classifier)
  return network
end

function modelTransformer:definePretraining(structure, l, options)
  -- TODO
  return model;
end

function modelTransformer:parametersDefault()
  self.initialize = nninit.xavier;
  self.nonLinearity = nn.RReLU;
  self.batchNormalize = true;
  self.kernelWidth = {};
  self.pretrain = false;
  self.padding = true;
  self.dropout = 0.5;
  self.cnn = '200,500,200';
  self.locnet = '30,60,30';
  self.locnet2 = '30,60,30';
  self.tra = true;
  self.rot = true;
  self.sca = true;
end

function modelTransformer:parametersRandom()
  -- All possible non-linearities
  self.distributions.nonLinearity = {nn.HardTanh, nn.HardShrink, nn.SoftShrink, nn.SoftMax, nn.SoftMin, nn.SoftPlus, nn.SoftSign, nn.LogSigmoid, nn.LogSoftMax, nn.Sigmoid, nn.Tanh, nn.ReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.LeakyReLU};
  self.distributions.initialize = {nninit.normal, nninit.uniform, nninit.xavier, nninit.kaiming, nninit.orthogonal, nninit.sparse};
end