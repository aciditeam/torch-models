-- Minimal harmonic test suite
-- Main definitions
-- 
-- Exports functions for the writing of simple music,
-- at the beat scale, with only 12 pitches (~ chromagrams)
-- The output tensors are 1-normalized for homogeneity.

local _ = require 'moses'
local music = require '../../music.lua/music.lua'

local M = {}

-- Declare individual pitches
-- Note names are taken only with the flat symbol 'b', no sharps.
-- 
-- For instance, M.pitch['Db'] is a 12-dimensional tensor with
-- all dimensions set to 0 except for the second one, which
-- has value 1.
function M.noteToTensor(note)
   local noteInt
   if _.isString(note) then
      noteInt = music.noteToInt(note)
   else
      noteInt = note
   end
   local noteIndex = noteInt%12 + 1
   local noteTensor = torch.zeros(1, 12)
   noteTensor:select(2, noteIndex):add(1)
   return noteTensor
end

M.compose = {}

-- Takes a table of note names and returns a normalized tensor
-- for the associated chord 
function M.compose.chordToTensor(notes)
   local chordTensor = torch.zeros(1, 12)
   for __, note in ipairs(notes) do
      chordTensor:add(M.noteToTensor(note))
   end
   return chordTensor:renorm(1, 1, 1)
end

-- Takes a table describing a sequence of chords and returns
-- the associated tensor, each step being a beat
-- 
-- Each element of the input table can be either:
--  * a string, describing a single pitch class
--  * a table of strings, describing a chord
function M.compose.sequence(sequenceTable)
   local sequenceDuration = #sequenceTable
   local sequence = torch.zeros(sequenceDuration, 12)
   for beat_i, beatContent in ipairs(sequenceTable) do
      local beatTensor
      if _.isTable(beatContent) then
	 -- current beat holds a chord
	 local chordTensor = M.compose.chordToTensor(beatContent)
	 beatTensor = chordTensor
      else
	 -- current beat holds a single note
	 local noteTensor = M.noteToTensor(beatContent)
	 beatTensor = noteTensor
      end
      sequence:select(1, beat_i):add(beatTensor)
   end
   return sequence
end

return M
