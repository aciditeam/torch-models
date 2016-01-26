--[[
  Convolutional LSTM for short term visual cell
  inputSize - number of input feature planes
  outputSize - number of output feature planes
  rho - recurrent sequence length
  kc  - convolutional filter size to convolve input
  km  - convolutional filter size to convolve cell; usually km > kc  
--]]
local _ = require 'moses'
require 'nn'
require 'dpnn'
require 'rnn'
require 'extracunn'

local ConvLSTM, parent = torch.class('nn.ConvLSTM', 'nn.AbstractRecurrent')

function ConvLSTM:__init(inputSize, outputSize, rho, kc, km, stride)
   parent.__init(self, rho or 10)
   self.inputSize = inputSize
   self.outputSize = outputSize
   self.kc = kc
   self.km = km
   self.padc = torch.floor(kc/2)
   self.padm = torch.floor(km/2)
   self.stride = stride or 1
   
   -- build the model
   self.recurrentModule = self:buildModel()
   -- make it work with nn.Container
   self.modules[1] = self.recurrentModule
   self.sharedClones[1] = self.recurrentModule 
   
   -- for output(0), cell(0) and gradCell(T)
   self.zeroTensor = torch.Tensor() 
   
   self.cells = {}
   self.gradCells = {}
end

-------------------------- factory methods -----------------------------
function ConvLSTM:buildGate()
   -- Note : Input is : {input(t), output(t-1), cell(t-1)}
   local gate = nn.Sequential()
   gate:add(nn.NarrowTable(1,2)) -- we don't need cell here
   local input2gate = nn.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate) 
   gate:add(para)
   gate:add(nn.CAddTable())
   gate:add(nn.Sigmoid())
   return gate
end

function ConvLSTM:buildInputGate()
   self.inputGate = self:buildGate()
   return self.inputGate
end

function ConvLSTM:buildForgetGate()
   self.forgetGate = self:buildGate()
   return self.forgetGate
end

function ConvLSTM:buildcellGate()
   -- Input is : {input(t), output(t-1), cell(t-1)}, but we only need {input(t), output(t-1)}
   local hidden = nn.Sequential()
   hidden:add(nn.NarrowTable(1,2))
   local input2gate = nn.SpatialConvolution(self.inputSize, self.outputSize, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)
   local output2gate = nn.SpatialConvolutionNoBias(self.outputSize, self.outputSize, self.km, self.km, self.stride, self.stride, self.padm, self.padm)
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate)
   hidden:add(para)
   hidden:add(nn.CAddTable())
   hidden:add(nn.Tanh())
   self.cellGate = hidden
   return hidden
end

function ConvLSTM:buildcell()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   self.inputGate = self:buildInputGate() 
   self.forgetGate = self:buildForgetGate()
   self.cellGate = self:buildcellGate()
   -- forget = forgetGate{input, output(t-1), cell(t-1)} * cell(t-1)
   local forget = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(self.forgetGate):add(nn.SelectTable(3))
   forget:add(concat)
   forget:add(nn.CMulTable())
   -- input = inputGate{input(t), output(t-1), cell(t-1)} * cellGate{input(t), output(t-1), cell(t-1)}
   local input = nn.Sequential()
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.cellGate)
   input:add(concat2)
   input:add(nn.CMulTable())
   -- cell(t) = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())
   self.cell = cell
   return cell
end   
   
function ConvLSTM:buildOutputGate()
   self.outputGate = self:buildGate()
   return self.outputGate
end

-- cell(t) = cell{input, output(t-1), cell(t-1)}
-- output(t) = outputGate{input, output(t-1)}*tanh(cell(t))
-- output of Model is table : {output(t), cell(t)} 
function ConvLSTM:buildModel()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   self.cell = self:buildcell()
   self.outputGate = self:buildOutputGate()
   -- assemble
   local concat = nn.ConcatTable()
   concat:add(nn.NarrowTable(1,2)):add(self.cell)
   local model = nn.Sequential()
   model:add(concat)
   -- output of concat is {{input(t), output(t-1)}, cell(t)}, 
   -- so flatten to {input(t), output(t-1), cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(3))
   cellAct:add(nn.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())
   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
   model:add(concat4)
   return model
end

------------------------- forward backward -----------------------------
function ConvLSTM:updateOutput(input)
   local prevOutput, prevCell
   
   if self.step == 1 then
      prevOutput = self.userPrevOutput or self.zeroTensor
      prevCell = self.userPrevCell or self.zeroTensor
      self.zeroTensor:resize(self.outputSize,input:size(2),input:size(3)):zero()
   else
      -- previous output and memory of this module
      prevOutput = self.output
      prevCell   = self.cell
   end
      
   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell
   if self.train ~= false then
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      -- the actual forward propagation
      output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
   end
   
   if self.train ~= false then
      local input_ = self.inputs[self.step]
      self.inputs[self.step] = self.copyInputs 
         and nn.rnn.recursiveCopy(input_, input) 
         or nn.rnn.recursiveSet(input_, input)     
   end
   
   self.outputs[self.step] = output
   self.cells[self.step] = cell
   
   self.output = output
   self.cell = cell
   
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradParametersAccumulated = false
   -- note that we don't return the cell, just the output
   return self.output
end

function ConvLSTM:backwardThroughTime(timeStep, rho)
   assert(self.step > 1, "expecting at least one updateOutput")
   self.gradInputs = {} -- used by Sequencer, Repeater
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   if self.fastBackward then
      for step=timeStep-1,math.max(stop,1),-1 do
         -- set the output/gradOutput states of current Module
         local recurrentModule = self:getStepModule(step)
         
         -- backward propagate through this step
         local gradOutput = self.gradOutputs[step]
         if self.gradPrevOutput then
            self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
            nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
            gradOutput = self._gradOutputs[step]
         end
         
         local scale = self.scales[step]
         local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
         local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
         local inputTable = {self.inputs[step], output, cell}
         local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
         local gradInputTable = recurrentModule:backward(inputTable, {gradOutput, gradCell}, scale)
         gradInput, self.gradPrevOutput, gradCell = unpack(gradInputTable)
         self.gradCells[step-1] = gradCell
         table.insert(self.gradInputs, 1, gradInput)
         if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
      end
      self.gradParametersAccumulated = true
      return gradInput
   else
      local gradInput = self:updateGradInputThroughTime()
      self:accGradParametersThroughTime()
      return gradInput
   end
end

function ConvLSTM:updateGradInputThroughTime(timeStep, rho)
   assert(self.step > 1, "expecting at least one updateOutput")
   self.gradInputs = {}
   local gradInput
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho

   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local gradOutput = self.gradOutputs[step]
      if self.gradPrevOutput then
         self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], self.gradPrevOutput)
         nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
         gradOutput = self._gradOutputs[step]
      end
      
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
      local inputTable = {self.inputs[step], output, cell}
      local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
      local gradInputTable = recurrentModule:updateGradInput(inputTable, {gradOutput, gradCell})
      gradInput, self.gradPrevOutput, gradCell = unpack(gradInputTable)
      self.gradCells[step-1] = gradCell
      table.insert(self.gradInputs, 1, gradInput)
      if self.userPrevOutput then self.userGradPrevOutput = self.gradPrevOutput end
   end
   
   return gradInput
end

function ConvLSTM:accGradParametersThroughTime(timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local scale = self.scales[step]
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
      local inputTable = {self.inputs[step], output, cell}
      local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]
      local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
      local gradOutputTable = {gradOutput, gradCell}
      recurrentModule:accGradParameters(inputTable, gradOutputTable, scale)
   end
   
   self.gradParametersAccumulated = true
   return gradInput
end

function ConvLSTM:accUpdateGradParametersThroughTime(lr, timeStep, rho)
   timeStep = timeStep or self.step
   local rho = math.min(rho or self.rho, timeStep-1)
   local stop = timeStep - rho
   
   for step=timeStep-1,math.max(stop,1),-1 do
      -- set the output/gradOutput states of current Module
      local recurrentModule = self:getStepModule(step)
      
      -- backward propagate through this step
      local scale = self.scales[step] 
      local output = (step == 1) and (self.userPrevOutput or self.zeroTensor) or self.outputs[step-1]
      local cell = (step == 1) and (self.userPrevCell or self.zeroTensor) or self.cells[step-1]
      local inputTable = {self.inputs[step], output, cell}
      local gradOutput = (step == self.step-1) and self.gradOutputs[step] or self._gradOutputs[step]
      local gradCell = (step == self.step-1) and (self.userNextGradCell or self.zeroTensor) or self.gradCells[step]
      local gradOutputTable = {self.gradOutputs[step], gradCell}
      recurrentModule:accUpdateGradParameters(inputTable, gradOutputTable, lr*scale)
   end
   
   return gradInput
end


function ConvLSTM:initBias(forgetBias, otherBias)
  local fBias = forgetBias or 1
  local oBias = otherBias or 0
  self.inputGate.modules[2].modules[1].bias:fill(oBias)
  --self.inputGate.modules[2].modules[2].bias:fill(oBias)
  self.outputGate.modules[2].modules[1].bias:fill(oBias)
  --self.outputGate.modules[2].modules[2].bias:fill(oBias)
  self.cellGate.modules[2].modules[1].bias:fill(oBias)
  --self.cellGate.modules[2].modules[2].bias:fill(oBias)
  self.forgetGate.modules[2].modules[1].bias:fill(fBias)
  --self.forgetGate.modules[2].modules[2].bias:fill(fBias)
end


--[[

require 'nn'
require 'unsupgpu'

-- Encoder
encoder = nn.Sequential()
local conv = nn.Sequential()
conv:add(nn.SpatialConvolution(opt.nFilters[1], opt.nFilters[2], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
local conv_new = require('weight-init')(conv, 'xavier')
encoder:add(conv_new)
encoder:add(nn.Tanh())
--encoder:add(nn.SmoothHuberPenalty(opt.nFilters[2], opt.constrWeight[1]))
--encoder:add(nn.L1Penalty(0.01)) 
encoder:add(nn.SpatialMaxPooling(2,2))

require 'nn'

-- Decoder, mirror of the encoder, but no non-linearity
-- first layer 
decoder = nn.Sequential()
--decoder:add(nn.SpatialUpSamplingNearest(2))
--decoder:add(nn.SpatialConvolution(opt.nFilters[3], opt.nFilters[2], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
--decoder:add(nn.Diag(opt.nFilters[2]))

-- second layer 

--decoder:add(nn.Dropout(0.5))
--decoder:add(nn.SpatialUnPooling(2))
decoder:add(nn.SpatialUpSamplingNearest(2)) 
local conv = nn.Sequential()
conv:add(nn.SpatialConvolution(opt.nFilters[2], opt.nFilters[1], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
local conv_new = require('weight-init')(conv, 'xavier')
decoder:add(conv_new)
--decoder:add(nn.SpatialUpSamplingNearest(2)) 
--decoder:add(nn.Diag(opt.nFilters[1]))

model = nn.Sequential()

-- add encoder
local seqe = nn.Sequencer(encoder)
--seqe:remember('both')
seqe:training()
model:add(seqe)

-- memory branch
local memory_branch = nn.Sequential()
local seq = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1],opt.nFiltersMemory[2], opt.nSeq, opt.kernelSize, opt.kernelSizeMemory, opt.stride))
seq:remember('both')
seq:training()
memory_branch:add(seq)
memory_branch:add(nn.SelectTable(opt.nSeq))
--memory_branch:add(nn.SelectTable(opt.nSeq))
--memory_branch:add(nn.L1Penalty(opt.constrWeight[2]))
memory_branch:add(flow)

-- keep last frame to apply optical flow on
local branch_up = nn.Sequential()
branch_up:add(nn.SelectTable(opt.nSeq))

-- transpose feature map for the sampler 
branch_up:add(nn.Transpose({1,3},{1,2}))

local concat = nn.ConcatTable()
concat:add(branch_up):add(memory_branch)
model:add(concat)

-- add sampler
model:add(nn.BilinearSamplerBHWD())
model:add(nn.Transpose({1,3},{2,3})) -- untranspose the result!!

-- add spatial decoder
model:add(decoder)

-- loss module: penalise difference of gradients
local gx = torch.Tensor(3,3):zero()
gx[2][1] = -1
gx[2][2] =  0
gx[2][3] =  1
gx = gx:cuda()
local gradx = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
gradx.weight:copy(gx)
gradx.bias:fill(0)

local gy = torch.Tensor(3,3):zero()
gy[1][2] = -1
gy[2][2] =  0
gy[3][2] =  1
gy = gy:cuda()
local grady = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
grady.weight:copy(gy)
grady.bias:fill(0)

local gradconcat = nn.ConcatTable()
gradconcat:add(gradx):add(grady)

gradloss = nn.Sequential()
gradloss:add(gradconcat)
gradloss:add(nn.JoinTable(1))

criterion = nn.MSECriterion()
--criterion.sizeAverage = false

-- move everything to gpu
model:cuda()
gradloss:cuda()
criterion:cuda()

for t = 1,opt.maxIter do
  --------------------------------------------------------------------
    -- progress
    iter = iter+1

    --------------------------------------------------------------------
    -- define eval closure
    local feval = function()
      local f = 0
 
      model:zeroGradParameters()

      inputTable = {}
      target  = torch.Tensor()--= torch.Tensor(opt.transf,opt.memorySizeH, opt.memorySizeW) 
      sample = datasetSeq[t]
      data = sample[1]
      for i = 1,data:size(1)-1 do
        table.insert(inputTable, data[i]:cuda())
      end
      target:resizeAs(data[1]):copy(data[data:size(1)])
    
      target = target:cuda()
      
      -- estimate f and gradients
      output = model:updateOutput(inputTable)
      gradtarget = gradloss:updateOutput(target):clone()
      gradoutput = gradloss:updateOutput(output)

      f = f + criterion:updateOutput(gradoutput,gradtarget)

      -- gradients
      local gradErrOutput = criterion:updateGradInput(gradoutput,gradtarget)
      local gradErrGrad = gradloss:updateGradInput(output,gradErrOutput)
           
      model:updateGradInput(inputTable,gradErrGrad)

      model:accGradParameters(inputTable, gradErrGrad)  

      grads:clamp(-opt.gradClip,opt.gradClip)
      return f, grads
    end
   
   
    if math.fmod(t,20000) == 0 then
      epoch = epoch + 1
      eta = opt.eta*math.pow(0.5,epoch/50)    
    end  

    rmspropconf = {learningRate = eta,
                  epsilon = 1e-5,
                  alpha = 0.9}

    _,fs = optim.rmsprop(feval, parameters, rmspropconf)

    err = err + fs[1]
    model:forget()
    --------------------------------------------------------------------
    -- compute statistics / report error
    if math.fmod(t , opt.nSeq) == 1 then
      print('==> iteration = ' .. t .. ', average loss = ' .. err/(opt.nSeq) .. ' lr '..eta ) -- err/opt.statInterval)
      err = 0
      if opt.save and math.fmod(t , opt.nSeq*1000) == 1 and t>1 then
        -- clean model before saving to save space
        --  model:forget()
        -- cleanupModel(model)         
        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
        torch.save(opt.dir .. '/rmspropconf_' .. t .. '.bin', rmspropconf)
      end
      
      if opt.display then
        _im1_ = image.display{image=inputTable[#inputTable-4]:squeeze(),win = _im1_, legend = 't-4'}
        _im2_ = image.display{image=inputTable[#inputTable-3]:squeeze(),win = _im2_, legend = 't-3'}
        _im3_ = image.display{image=inputTable[#inputTable-2]:squeeze(),win = _im3_, legend = 't-2'}
        _im4_ = image.display{image=inputTable[#inputTable-1]:squeeze(),win = _im4_, legend = 't-1'}
        _im5_ = image.display{image=inputTable[#inputTable]:squeeze(),win = _im5_, legend = 't'}
        _im6_ = image.display{image=target:squeeze(),win = _im6_, legend = 'Target'}
        _im7_ = image.display{image=output:squeeze(),win = _im7_, legend = 'Output'}

        local imflow = flow2colour(optical_flow)
        _im8_ = image.display{image=imflow,win=_im8_,legend='Flow'}
          
        print (' ==== Displaying weights ==== ')
        -- get weights
        eweight = model.modules[1].module.modules[1].modules[1].modules[1].weight
        dweight = model.modules[5].modules[2].modules[1].weight
        dweight_cpu = dweight:view(opt.nFilters[2], opt.kernelSize, opt.kernelSize)
        eweight_cpu = eweight:view(opt.nFilters[2], opt.kernelSize, opt.kernelSize)
        -- render filters
        dd = image.toDisplayTensor{input=dweight_cpu,
                                   padding=2,
                                   nrow=math.floor(math.sqrt(opt.nFilters[2])),
                                   symmetric=true}
        de = image.toDisplayTensor{input=eweight_cpu,
                                   padding=2,
                                   nrow=math.floor(math.sqrt(opt.nFilters[2])),
                                   symmetric=true}

        -- live display
        if opt.display then
           _win1_ = image.display{image=dd, win=_win1_, legend='Decoder filters', zoom=8}
           _win2_ = image.display{image=de, win=_win2_, legend='Encoder filters', zoom=8}
        end
      end  
    end
  end

]]--