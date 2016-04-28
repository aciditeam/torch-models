require 'torch'
require 'nn'
require 'rnn'
require 'importTSDataset'
require 'mainVisualize'



print("yop");

local nninit = require 'nninit'

testi = nn.Linear(100, 200):init('weight', nninit.kaiming);

plot = Plot():histogram(testi.weight:reshape(100 * 200)):draw()
--plot = Plot():histogram(testi.gradParams:reshape(100 * 200)):draw()
--plot:save('check.html');
--os.execute('open check.html');

plots = {}
-- segment plots
x1=torch.randn(10)
y1=torch.randn(10)
plots[1] = Plot():segment(x1, y1, x1+10,y1+10, 'red','demo'):title('Segment Plot Demo')
-- quiver plots
U = torch.randn(3,3):mul(100)
V = torch.randn(3,3):mul(100)
plots[2] = Plot():quiver(U,V,'red',''):title('Quiver Plot Demo')
--html = multiPlot(plots);

--sidvb:poy()

--plot:save('out2.html')
--os.execute('open out2.html')
-- quads/rectangles
--x1=torch.randn(10)
--y1=torch.randn(10)
--plot = Plot():quad(x1,y1,x1+1,y1+1,'red',''):draw()

--[[
-- histogram
plot = Plot():histogram(torch.randn(10000)):draw()
local t = torch.Tensor
local y = t(10)
local x = t(y:size()):zero()
local labels = {}
for i = 1, 10 do
    y[i] = i
    labels[i] = tostring(i)
end

Plot()
  :circle(x, y, 'red', nil, {foo=labels})
  :hover_tool({{'xy', '@x @y'}, {'foo', '@foo'}})
  :draw()
  
local t = torch.Tensor
local y = t(10)
local x = t(y:size()):zero()
local labels = {}
for i = 1, 10 do
    y[i] = i
    labels[i] = tostring(i)
end

Plot():gscatter(x, y)
  :text(x, y, labels, y, 'black')
  :triangle(x, y, 'blue')
  :draw()

]]--

baseDir = '/Users/esling/Dropbox/TS_Datasets';
setList = {'ArrowHead'};
local sets = import_data(baseDir, setList);
--testiSeries = {};
--i = 1;
--for v,n in ipairs({16,32,50,64,100,128,200,256,333,512,1024,2048,4096,8192}) do
--  testiSeries[i] = tensorResampling(sets['Beef']['TRAIN'].data[5]:double(), n, 'gaussian');
--  testiSeries[i]:resize(n);
--  i = i + 1;
--end
series = sets['ArrowHead']['TRAIN'].data[{4, {}}]:reshape(1, 251);
print(series);
loc = torch.Tensor{0.6, -0.5};
print(loc);
tg = nn.TemporalGlimpse(16, 8, 1.5);
testiSeries = tg:forward{series, loc};
print(testiSeries);
--tg:backward({series, loc}, testiSeries)
print(testiSeries)
plotMultipleTimeSeries(testiSeries[1]);
zeiury:zeoiru();
inputFrameSize = 1;
outputFrameSize = 10;
kW = 16; dW = 1;
model = nn.Sequential();
model:add(nn.Reshape(128, 1));
model:add(nn.Padding(1, -kW/2));
model:add(nn.Padding(1, kW/2));
model:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kW, dW));
model:add(nn.ReLU());
model:add(nn.TemporalConvolution(outputFrameSize, outputFrameSize, 3, 1));
model:add(nn.ReLU());
model:add(nn.TemporalMaxPooling(2, 2));
model:add(nn.Dropout(0.25));
--print(model:get(2).weight)
--model:get(3).weight = torch.zeros(20, 16);
--for i = 1,16 do
--  model:get(3).weight[{i, i}] = 1;
--end
--model.bias = torch.zeros(20, 1);
x = sets['ArrowHead']["TRAIN"].data[1]
print(x);



testiPote = nn.SlidingWindow(1, 16, 1, 1e9, false);
check = testiPote:forward(x);
--plotMultipleTimeSeries(check);
--plotMultipleTimeSeries(sets['ArrowHead']["TRAIN"].data)
--for i = 1,#check do
--  plotTS(check[i]);
--end

sidvb:poy()

local r = nn.Recurrent(
   100, 
   nn.Linear(1, 100), 
   nn.Sigmoid(), 
   rho
)

local rnn = nn.Sequential()
   :add(r)
   :add(nn.Linear(100, 10))
   :add(nn.LogSoftMax())

-- wrap the non-recurrent module (Sequential) in Recursor.
-- This makes it a recurrent module
-- i.e. Recursor is an AbstractRecurrent instance
rnn = nn.Recursor(rnn, rho)

finalNN = nn.Sequential():add(rnn):add(nn.Reshape(1280)):add(nn.Linear(1280, 100));

inputSize = 10;
hiddenSize = 512;
outputSize = 3;

lstm = nn.Sequential() 
      :add(nn.Linear(inputSize, hiddenSize))
      :add(nn.LSTM(hiddenSize, hiddenSize))
      :add(nn.LSTM(hiddenSize, hiddenSize))
      :add(nn.Linear(hiddenSize, outputSize))
      :add(nn.LogSoftMax()) 
      
lstm = nn.Recursor(lstm);
   
print("YEOW")
print(lstm:forward(x))

print("SHIST");
yop:size();

--require 'rnn'
require 'torch'
require 'nn'
require 'rnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Simple LSTM example for the RNN library')
cmd:text()
cmd:text('Options')
cmd:option('-use_saved',false,'Use previously saved inputs and trained network instead of new')
cmd:option('-cuda',false,'Run on CUDA-enabled GPU instead of CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.cuda then
   require 'cunn'
end         

-- Keep the input layer small so the model trains / converges quickly while training
local inputSize = 2
-- Larger numbers here mean more complex problems can be solved, but can also over-fit. 256 works well for now
local hiddenSize = 512
-- We want the network to classify the inputs using a one-hot representation of the outputs
local outputSize = 3

-- the dataset size is the total number of examples we want to present to the LSTM 
local dsSize=2000

-- We present the dataset to the network in batches where batchSize << dsSize
local batchSize=64

--And seqLength is the length of each sequence, i.e. the number of "events" we want to pass to the LSTM
--to make up a single example. I'd like this to be dynamic ideally for the YOOCHOOSE dataset..
local seqLength=128

-- number of target classes or labels, needs to be the same as outputSize above
-- or we get the dreaded "ClassNLLCriterion.lua:46: Assertion `cur_target >= 0 && cur_target < n_classes' failed. "
local nClass = 3

function build_data()
   local inputs = {}
   local targets = {}
   --Use previously created and saved data
   if opt.use_saved then
      inputs = torch.load('training.t7')
      targets = torch.load('targets.t7')
      rnn = torch.load('trained-model.t7')
   else
      for i = 1, dsSize do
         -- populate both tables to get ready for training
         local input = torch.randn(batchSize,inputSize)
         local target = torch.LongTensor(batchSize):random(1,nClass)
         if opt.cuda then
            input = input:float():cuda()
            target = target:float():cuda()
         end
         table.insert(inputs, input)
         table.insert(targets, target)
      end
   end
   return inputs, targets
end

function build_network(inputSize, hiddenSize, outputSize)
   if opt.use_saved then
      rnn = torch.load('trained-model.t7')
   else
      rnn = nn.Sequential() 
      :add(nn.Linear(inputSize, hiddenSize))
      :add(nn.LSTM(hiddenSize, hiddenSize))
      :add(nn.LSTM(hiddenSize, hiddenSize))
      :add(nn.Linear(hiddenSize, outputSize))
      :add(nn.LogSoftMax())
      -- wrap this in a Sequencer such that we can forward/backward 
      -- entire sequences of length seqLength at once
      rnn = nn.Sequencer(rnn)
      if opt.cuda then
         rnn:cuda()
      end
   end
   return rnn
end

function save(inputs, targets, rnn)
   -- Save out the tensors we created and the model itself so we can load it back in
   -- if -use_saved is set to true
   torch.save('training.t7', inputs)
   torch.save('targets.t7', targets)
   torch.save('trained-model.t7', rnn)
end

-- two tables to hold the *full* dataset input and target tensors
local inputs, targets = build_data()
local rnn = build_network(inputSize, hiddenSize, outputSize)

-- Decorate the regular nn Criterion with a SequencerCriterion as this simplifies training quite a bit
-- SequencerCriterion requires tables as input, and this affects the code we have to write inside the training for loop
local crit = nn.ClassNLLCriterion()
local seqC = nn.SequencerCriterion(crit)
if opt.cuda then
   crit:cuda()
   seqC:cuda()
end

-- Now let's train our network on the small, fake dataset we generated earlier
rnn:training()

print('Start training')
--Feed our LSTM the dsSize examples in total, broken into batchSize chunks
for numEpochs=0,500,1 do
   local err = 0
   local start =torch.tic() 

   for offset=1,dsSize,batchSize+seqLength do
      local batchInputs = {}
      local batchTargets = {}
      -- We need to get a subset (of size batchSize) of the inputs and targets tables

      -- start needs to be "2" and end "batchSize-1" to correctly index
      -- all of the examples in the "inputs" and "targets" tables
      for i = 2, batchSize+seqLength-1,1 do
         table.insert(batchInputs, inputs[offset+i])
         table.insert(batchTargets, targets[offset+i])
      end
      --local currT = torch.toc(start)
      -- print('Created ds in ', currT .. 's')
      -- start = torch.tic()
      local out = rnn:forward(batchInputs)
      -- currT = torch.toc(start)
      -- print('rnn:forward in ', currT .. 's')
      -- start = torch.tic()
      
      print(out);
      print(batchInputs);
      print(batchTargets[1]:reshape(1, 64));
      print(batchTargets[2]:reshape(1, 64));
      print(batchTargets[3]:reshape(1, 64));
      --print(torch.cat(batchTargets[1], batchTargets[2], batchTargets[3]));
      print(seqC:forward(out, batchTargets));
      yoefefp:size();
      -- currT = torch.toc(start)
      -- print('seqC:forward in ', currT .. 's')
      -- start = torch.tic()
      gradOut = seqC:backward(out, batchTargets)
      -- currT = torch.toc(start)
      -- print('seqC:backward in ', currT .. 's')
      -- start = torch.tic()
      rnn:backward(batchInputs, gradOut)
      -- currT = torch.toc(start)
      -- print('rnn:backward in ', currT .. 's')
      -- start = torch.tic()
      --We update params at the end of each batch
      rnn:updateParameters(0.05)
      rnn:zeroGradParameters()
      --local currT = torch.toc(start)
      --print(collectgarbage("count") .. 'MB ', currT .. ' s')
   end
   local currT = torch.toc(start)
   print('loss', err/dsSize .. ' in ', currT .. ' s')
end