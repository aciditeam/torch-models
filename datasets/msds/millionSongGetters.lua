----
-- Million Song Dataset Getter functions
--
-- TODO:
--  * Add missing functions.
--  * Add support for indexing in aggregate files.
----
-- 
-- Adapted from
-- Thierry Bertin-Mahieux (2010) Columbia University
-- tb2332@columbia.edu
-- 
-- 
-- This code contains a set of getters functions to access the fields
-- from an HDF5 song file (regular file with one song or
-- aggregate / summary file with many songs)
-- 
-- This is part of the Million Song Dataset project from
-- LabROSA (Columbia University) and The Echo Nest.
-- 
-- 
-- Copyright 2010, Thierry Bertin-Mahieux
-- 
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
-- 
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details.
-- 
-- You should have received a copy of the GNU General Public License
-- along with this program.  If not, see <http://www.gnu.org/licenses/>.

local hdf5 = require 'hdf5'

torch.setdefaulttensortype('torch.FloatTensor')

local getters = {}

-- Segment aligned features

function getters.open_h5_file_read(h5filename)
   -- Open an existing H5 in read mode.
   -- Same function as in hdf5_utils, here so we avoid one import
   local readFile = hdf5.open(h5filename, 'r')
   return readFile
end

function getters.get_segments_start(h5)
   -- Get segments start array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/segments_start';
   return h5:read(dataPath):all()
end

function getters.get_segments_confidence(h5)
   -- Get segments confidence array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/segments_confidence';
   return h5:read(dataPath):all()
end

function getters.get_segments_pitches(h5)
   -- Get segments pitches array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/segments_pitches';
   return h5:read(dataPath):all()
end

function getters.get_segments_timbre(h5)
   -- Get segments timbre array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/segments_timbre';
   return h5:read(dataPath):all()
end

function getters.get_segments_loudness_max(h5)
   -- Get segments loudness max array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/segments_loudness_max';
   return h5:read(dataPath):all()
end

function getters.get_segments_loudness_max_time(h5)
   -- Get segments loudness max time array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/segments_loudness_max_time';
   return h5:read(dataPath):all()
end

function getters.get_segments_loudness_start(h5)
   -- Get segments loudness start array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/segments_loudness_start';
   return h5:read(dataPath):all()
end

function getters.get_sections_start(h5)
   -- Get sections start array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/section_start';
   return h5:read(dataPath):all()
end

function getters.get_sections_confidence(h5)
   -- Get sections confidence array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/sections_confidence';
   return h5:read(dataPath):all()
end

-- Beat alignment features

function getters.get_beats_start(h5)
   -- Get beats start array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/beats_start';
   return h5:read(dataPath):all()
end

function getters.get_beats_confidence(h5)
   -- Get beats confidence array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/beats_confidence';
   return h5:read(dataPath):all()
end

function getters.get_bars_start(h5)
   -- Get bars start array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/bars_start';
   return h5:read(dataPath):all()
end

function getters.get_bars_confidence(h5)
   -- Get bars start confidence array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/bars_confidence';
   return h5:read(dataPath):all()
end

function getters.get_tatums_start(h5)
   -- Get tatums start array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/tatums_start';
   return h5:read(dataPath):all()
end

function getters.get_tatums_confidence(h5)
   -- Get tatums confidence array.
   -- Takes care of the proper indexing if we are in aggregate file.
   -- By default, return the array for the first song in the h5 file.
   local dataPath = 'analysis/tatums_confidence';
   return h5:read(dataPath):all()
end

return getters
