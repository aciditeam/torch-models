-- Minimal harmonic test suite
-- Functions for harmonic testing of prediction models

require 'nn'
require '../../moduleSlidingWindow.lua'
require '../../modelLSTM.lua'
local ts_init = require '../../TSInitialize.lua'
local _ = require 'moses'

cmd = torch.CmdLine()
cmd:option('--loadModel', '', 'path to previously trained model')
cmd:option('--useCuda', false, 'whether to activate CUDA')

cmd_params = cmd:parse(arg)

local music = require '../../music.lua/music.lua'
local harmonic = require './harmonic.lua'

local M = {}

local criterion = nn.MSECriterion()

local useCuda = cmd_params.useCuda
local nInputs = 128  -- Duration of input sequences for the considered models
local paddingValue = 0
local featSize = 12

local function makePrediction(model, inputSequence, predictionLength, repeatInput)
   local repeatInput = repeatInput or true

   local function pad(sequence)
      local sequenceDuration = sequence:size(1)
      
      if sequenceDuration < nInputs then
	 if repeatInput and nInputs > 2*sequenceDuration then
	    nRepeats = math.floor(nInputs / sequenceDuration)
	    local sizes = torch.LongStorage({nRepeats, 1})
	    return pad(inputSequence:repeatTensor(sizes))
	 else
	    -- Sequence is shorter than the input size: add silence at the beginning
	    local deltaDuration = nInputs - sequenceDuration
	    
	    local padding = torch.Tensor(deltaDuration, featSize)
	    padding:fill(paddingValue)
	    
	    local paddedSequence = padding:cat(sequence, 1)
	    return paddedSequence
	 end
      else
	 return sequence
      end
   end

   local function addPredictionShift(sequence, newStep)
      local sequenceOut = sequence:narrow(1, 2, nInputs-1)
      return sequenceOut:cat(newStep, 1)
   end
   
   local inputSequence = pad(inputSequence):view(nInputs, 1, featSize)
   if options.cuda then inputSequence = inputSequence:cuda() end
   print(inputSequence)
   local output = model:forward(inputSequence)
   
   for i=2, predictionLength do
      local inputAcc = addPredictionShift(inputSequence, output)
      print(inputAcc)
      
      local newPredictionStep = model:forward(inputAcc)
      print(newPredictionStep)
      output = output:cat(newPredictionStep, 1)
   end

   return output:view(predictionLength, featSize)
end

function M.testModel(model, inputSequence, target, testDescription,
		     predictionLength)
   local predictionLength = predictionLength or 1
   
   local output = makePrediction(model, inputSequence, predictionLength, true)
   if useCuda then target = target:cuda() end
   local err = criterion:forward(output, target)

   print(testDescription)
   print('On input sequence: ')
   print(inputSequence)
   print('With expected output: ')
   print(target)
   print("Model's prediction: ")
   print(output)
   print("L2-error: ")
   print(err)
end


local function makeSequenceScaleDegrees(degrees, scale)
   local makeDegreeToChord = function(__, degree)
      return music.diatonicChord(degree, scale)
   end
   
   return harmonic.compose.sequence(_.map(degrees,
					  makeDegreeToChord))
end

-- Test alternate
-- Test simple alternating chords
local EMinScale = music.scale('E', music.scales.minor)

-- This sequence is 4*CMaj followed by 4*GMaj 
local alternatingChords = makeSequenceScaleDegrees(
   {1, 5, 1, 5, },
   EMinScale)

local alternateTest_simple_description = "Simple, 'abababab' -> 'a', 1-step memory test"
local alternateTest_simple_target = makeSequenceScaleDegrees(
   {1, }, EMinScale)
function M.makeAlternateTest_simple(model)
   M.testModel(model, alternatingChords, alternateTest_simple_target,
	       alternateTest_simple_description)
end

local alternateTest_double_description = "Double prediction, 'abababab' - 'ab', " ..
   "1-step memory test"
local alternateTest_double_target = makeSequenceScaleDegrees(
   {1, 5}, EMinScale)
local predictionLength = 2
function M.makeAlternateTest_double(model)
   M.testModel(model, alternatingChords, alternateTest_double_target,
	       alternateTest_double_description, predictionLength)
end

local alternateTest_triple_description = "Triple prediction, 'abababab' - 'aba', " ..
   "1-step memory test"
local alternateTest_triple_target = makeSequenceScaleDegrees(
   {1, 5, 1}, EMinScale)
local predictionLength = 3
function M.makeAlternateTest_triple(model)
   M.testModel(model, alternatingChords, alternateTest_triple_target,
	       alternateTest_triple_description, predictionLength)
end

-- Test chords looping
-- Test the model to repeat a loop of 4 succesive chords
-- followed by another 4 successive equal chords 
local CMajScale = music.scale('C', music.scales.major)

-- This sequence is 4*CMaj followed by 4*GMaj 
local chordsLoop = makeSequenceScaleDegrees({1, 1, 1, 1,
					     5, 5, 5, 5, 
					     1, 1, 1, }, CMajScale)

local loopTest_simple_description = "Model should predict to repeat last CMaj"
local loopTest_simple_target = makeSequenceScaleDegrees({1}, CMajScale)
function M.makeLoopTest_simple(model)
   M.testModel(model, chordsLoop, loopTest_simple_target,
	       loopTest_simple_description)
end

local loopTest_double_description = "Perform two successive prediction, " ..
   "\nModel should loop to the 2nd series of chords, i.e. output a CMaj then a GMaj"
local loopTest_double_target = makeSequenceScaleDegrees({1, 5}, CMajScale)
local predictionLength = 2
function M.makeLoopTest_double(model)
   M.testModel(model, chordsLoop, loopTest_double_target,
	       loopTest_double_description, predictionLength)
end

-- Expects G as continuation
local majorSequence = harmonic.compose.sequence(
   {'C', 'D', 'E', 'F'})
local majorSequence_target = harmonic.noteToTensor('G')


-- Expects Bb as continuation
local minorSequence = harmonic.compose.sequence(
   {'D', 'E', 'F', 'Ab'})
local minorSequence_target = harmonic.noteToTensor('Bb')

-- Test on a blues grid
local AMajScale = music.scale('A', music.scales.major)
local seventhChordDegree = function(degree, scale)
   return music.diatonicChord(degree, scale, "7") end
local A7 = seventhChordDegree(1, AMajScale)
local D7 = seventhChordDegree(4, AMajScale)
local E7 = seventhChordDegree(5, AMajScale)

-- local ABluesGrid = harmonic.compose.sequence(
--    {A7, A7, A7, A7,
--     D7, D7, A7, A7,
--     E7, D7, A7, E7, })

local degreesBluesGrid = {1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 5}
local ABluesGrid = makeSequenceScaleDegrees(degreesBluesGrid, AMajScale)

local bluesTest_simple_description = "Model should predict start of new grid, i.e. degree I, i.e. A"
local bluesTest_simple_target = makeSequenceScaleDegrees({1}, CMajScale)
function M.makeBluesTest_simple(model)
   M.testModel(model, ABluesGrid, bluesTest_simple_target,
	       bluesTest_simple_description)
end

-- Define or load a model
local options = ts_init.get_options(useCuda)
options.predict = true
options.cuda = useCuda

ts_init.set_cuda(options)

model = torch.load(cmd_params.loadModel)

model:evaluate()

for i=1,10000 do
   print(model:forward(torch.rand(128, 12):cuda()))
end

if options.cuda then model:cuda(); criterion:cuda() end

M.makeBluesTest_simple(model)
M.makeAlternateTest_double(model)

return M
