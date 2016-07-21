-- Sliding iterator testing function

local iterator = require '../slidingIterator.lua'
local _ = require 'moses'

local function append(mainTable, elems)
   for i=1,#elems do
      table.insert(mainTable, elems[i])
   end
end

local function isequal(table1, table2)
   if #table1 ~= #table2 then return false end
   for i=1,#table1 do
      if table1[i] ~= table2[i] then return false end
   end
   return true
end

local no_overlap = true
local verbose = false

for windowSize = 1, 5 do
   print('Testing windowSize: ' .. windowSize)
   for windowStep = 1, 5 do
      print('\tTesting windowStep: ' .. windowStep)
      for last_elem =1,100 do
	 local elems = _.range(1, last_elem)
	 if verbose then
	    print('Input elements: ')
	    print(elems)
	 end
	 
	 local f = function(elems) return elems end
	 
	 local recombine = {}

	 if verbose then print('\nStarting iterator') end
	 for window, cur_start, cur_end in iterator.foreach_windowed(
	    f, elems, windowSize, windowStep, no_overlap, verbose) do
	    if verbose then
	       print('\nWindow content: ')
	       print(window)
	       print('Position: ' .. '\n\tStart: ' .. cur_start .. '\n\tEnd: ' ..
			cur_end .. '\n')
	    end
	    
	    append(recombine, window)
	 end
	 
	 if verbose then print('End.') end
	 assert(isequal(recombine, elems))
      end
   end
end
