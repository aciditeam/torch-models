----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Siamese network
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'torch'
require 'nn'

modelSiamese = {}

function modelSiamese.defineSiameseConvolutional(libs)
    local SpatialConvolution = libs['SpatialConvolution']
    local SpatialMaxPooling = libs['SpatialMaxPooling']
    local ReLU = libs['ReLU']
    --Encoder/Embedding
    --Input dims are 28x28          NOTE: change dims as inputs are 32x32 -- Need to do this 
    encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolution(1, 20, 5, 5)) -- 1 input image channel, 20 output channels, 5x5 convolution kernel (each feature map is 28x28)
    encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- max pooling with kernal 2x2 and a stride of 2 in each direction (feature maps are 14x14)
    encoder:add(nn.SpatialConvolution(20, 50, 5, 5)) -- 20 input feature maps and output 50, 5x5 convolution kernel (feature maps are 10x10) 
    encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- max pooling (feature maps are 5x5)

    encoder:add(nn.View(50*5*5)) --reshapes to view data at 50x4x4
    encoder:add(nn.Linear(50*5*5, 500))
    encoder:add(nn.ReLU())
    encoder:add(nn.Linear(500, 10))
    --encoding layer - go from 10 class out to 2-dimensional encoding
    encoder:add(nn.Linear(10, 2))

    --The siamese model
    siamese_encoder = nn.ParallelTable()
    siamese_encoder:add(encoder)
    siamese_encoder:add(encoder:clone('weight','bias', 'gradWeight','gradBias')) --clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias


    --The siamese model (inputs will be Tensors of shape (2, channel, height, width))
    model = nn.Sequential()
    model:add(nn.SplitTable(1)) -- split input tensor along the rows (1st dimension) to table for input to ParallelTable
    model:add(siamese_encoder)
    model:add(nn.PairwiseDistance(2)) --L2 pariwise distance

    margin = 1
    criterion = nn.HingeEmbeddingCriterion(margin)
    return model
end

function modelSiamese.defineSiamese(libs)
  -- imagine we have one network we are interested in, it is called "p1_mlp"
  p1_mlp = nn.Sequential(); p1_mlp:add(nn.Linear(5, 2))
  -- But we want to push examples towards or away from each other so we make another copy
  -- of it called p2_mlp; this *shares* the same weights via the set command, but has its
  -- own set of temporary gradient storage that's why we create it again (so that the gradients
  -- of the pair don't wipe each other)
  p2_mlp = nn.Sequential(); p2_mlp:add(nn.Linear(5, 2))
  p2_mlp:get(1).weight:set(p1_mlp:get(1).weight)
  p2_mlp:get(1).bias:set(p1_mlp:get(1).bias)
  -- we make a parallel table that takes a pair of examples as input.
  -- They both go through the same (cloned) mlp
  prl = nn.ParallelTable()
  prl:add(p1_mlp)
  prl:add(p2_mlp)
  -- now we define our top level network that takes this parallel table
  -- and computes the pairwise distance betweem the pair of outputs
  mlp = nn.Sequential()
  mlp:add(prl)
  mlp:add(nn.PairwiseDistance(1))
  -- and a criterion for pushing together or pulling apart pairs
  crit = nn.HingeEmbeddingCriterion(1)
  -- lets make two example vectors
  x = torch.rand(5)
  y = torch.rand(5)
  -- Use a typical generic gradient update function
  function gradUpdate(mlp, x, y, criterion, learningRate)
    local pred = mlp:forward(x)
    local err = criterion:forward(pred, y)
    local gradCriterion = criterion:backward(pred, y)
    mlp:zeroGradParameters()
    mlp:backward(x, gradCriterion)
    mlp:updateParameters(learningRate)
  end
  -- push the pair x and y together, notice how then the distance between them given
  -- by print(mlp:forward({x, y})[1]) gets smaller
  for i = 1, 10 do
    gradUpdate(mlp, {x, y}, 1, crit, 0.01)
    print(mlp:forward({x, y})[1])
  end
  -- pull apart the pair x and y, notice how then the distance between them given
  -- by print(mlp:forward({x, y})[1]) gets larger
  for i = 1, 10 do
    gradUpdate(mlp, {x, y}, -1, crit, 0.01)
    print(mlp:forward({x, y})[1])
  end
end

--[[

function train(data)
    local saved_criterion = false;
    for i = 1, params.max_epochs do
        --add random shuffling here
        train_one_epoch(data)

        if params.snapshot_epoch > 0 and (epoch % params.snapshot_epoch) == 0 then -- epoch is global (gotta love lua :p)
            local filename = paths.concat(params.snapshot_dir, "snapshot_epoch_" .. epoch .. ".net")
            os.execute('mkdir -p ' .. sys.dirname(filename))
            torch.save(filename, model)        
            --must save std, mean and criterion?
            if not saved_criterion then
                local criterion_filename = paths.concat(params.snapshot_dir, "_criterion.net")
                torch.save(criterion_filename, criterion)
                local dataset_attributes_filename = paths.concat(params.snapshot_dir, "_dataset.params")
                dataset_attributes = {}
                dataset_attributes.mean = data.mean
                dataset_attributes.std = data.std
                torch.save(dataset_attributes_filename, dataset_attributes)
            end
        end
    end
end

function train_one_epoch(dataset)

    local time = sys.clock()
    --train one epoch of the dataset

    for mini_batch_start = 1, dataset:size(), batch_size do --for each mini-batch
        
        local inputs = {}
        local labels = {}
        --create a mini_batch
        for i = mini_batch_start, math.min(mini_batch_start + batch_size - 1, dataset:size()) do 
            local input = dataset[i][1]:clone() -- the tensor containing two images 
            local label = dataset[i][2] -- +/- 1
            table.insert(inputs, input)
            table.insert(labels, label)
        end
        --create a closure to evaluate df/dX where x are the model parameters at a given point
        --and df/dx is the gradient of the loss wrt to thes parameters

        local func_eval = 
        function(x)
                --update the model parameters (copy x in to parameters)
                if x ~= parameters then
                    parameters:copy(x) 
                end

                grad_parameters:zero() --reset gradients

                local avg_error = 0 -- the average error of all criterion outs

                --evaluate for complete mini_batch
                for i = 1, #inputs do
                    local output = model:forward(inputs[i])

                    local err = criterion:forward(output, labels[i])
                    avg_error = avg_error + err

                    --estimate dLoss/dW
                    local dloss_dout = criterion:backward(output, labels[i])
                    model:backward(inputs[i], dloss_dout)
                end

                grad_parameters:div(#inputs);
                avg_error = avg_error / #inputs;

                return avg_error, grad_parameters
        end


        config = {learningRate = params.learning_rate, momentum = params.momentum}

        --This function updates the global parameters variable (which is a view on the models parameters)
        optim.sgd(func_eval, parameters, config)
        
        xlua.progress(mini_batch_start, dataset:size()) --display progress
    end

    -- time taken
    time = sys.clock() - time
    print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/dataset:size())*1000) .. 'ms')
    epoch = epoch + 1
end


-- Train a ranking function so that mlp:forward({x, y}, {x, z}) returns a number
-- which indicates whether x is better matched with y or z (larger score = better match), or vice versa.

mlp1 = nn.Linear(5, 10)
mlp2 = mlp1:clone('weight', 'bias')

prl = nn.ParallelTable();
prl:add(mlp1); prl:add(mlp2)

mlp1 = nn.Sequential()
mlp1:add(prl)
mlp1:add(nn.DotProduct())

mlp2 = mlp1:clone('weight', 'bias')

mlp = nn.Sequential()
prla = nn.ParallelTable()
prla:add(mlp1)
prla:add(mlp2)
mlp:add(prla)

x = torch.rand(5);
y = torch.rand(5)
z = torch.rand(5)


print(mlp1:forward{x, x})
print(mlp1:forward{x, y})
print(mlp1:forward{y, y})


crit = nn.MarginRankingCriterion(1);

-- Use a typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

inp = {{x, y}, {x, z}}

math.randomseed(1)

-- make the pair x and y have a larger dot product than x and z

for i = 1, 100 do
   gradUpdate(mlp, inp, 1, crit, 0.05)
   o1 = mlp1:forward{x, y}[1];
   o2 = mlp2:forward{x, z}[1];
   o = crit:forward(mlp:forward{{x, y}, {x, z}}, 1)
   print(o1, o2, o)
end

print "________________**"

-- make the pair x and z have a larger dot product than x and y

for i = 1, 100 do
   gradUpdate(mlp, inp, -1, crit, 0.05)
   o1 = mlp1:forward{x, y}[1];
   o2 = mlp2:forward{x, z}[1];
   o = crit:forward(mlp:forward{{x, y}, {x, z}}, -1)
   print(o1, o2, o)
end

--------------------------------------------------------------------------------
-- TripletEmbeddingCriterion
--------------------------------------------------------------------------------
-- Alfredo Canziani, Apr/May 15
--------------------------------------------------------------------------------

local TripletEmbeddingCriterion, parent = torch.class('nn.TripletEmbeddingCriterion', 'nn.Criterion')

function TripletEmbeddingCriterion:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 0.2
   self.Li = torch.Tensor()
   self.gradInput = {}
end

function TripletEmbeddingCriterion:updateOutput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   self.Li:resize(N)
   for i = 1, N do
      self.Li[i] = math.max(0, (a[i]-p[i])*(a[i]-p[i])+self.alpha-(a[i]-n[i])*(a[i]-n[i]))
      --print(self.Li[i])
   end
   self.output = self.Li:sum() / N
   return self.output
end

function TripletEmbeddingCriterion:updateGradInput(input)
   local a = input[1] -- ancor
   local p = input[2] -- positive
   local n = input[3] -- negative
   local N = a:size(1)
   if torch.type(a) == 'torch.CudaTensor' then -- if buggy CUDA API
      self.gradInput[1] = (n - p):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
      self.gradInput[2] = (p - a):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
      self.gradInput[3] = (a - n):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
   else -- otherwise
      self.gradInput[1] = self.Li:gt(0):diag():type(a:type()) * (n - p) * 2/N
      self.gradInput[2] = self.Li:gt(0):diag():type(a:type()) * (p - a) * 2/N
      self.gradInput[3] = self.Li:gt(0):diag():type(a:type()) * (a - n) * 2/N
   end
   return self.gradInput
end


]]--