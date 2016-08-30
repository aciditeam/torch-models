-- Sliding window iterator on set of elements

local M = {}

--
function M.foreach_windowed(f, elems, windowSize, windowStep, no_overlap, verbose)
   local elemsNum = #elems
   if elemsNum == 0 then return
      function() return nil end
   end
   
   local windowSize = math.min(windowSize, elemsNum)
   local windowStep = math.max(windowStep or math.floor(windowSize / 10), 1)
   -- Restrict step to window size to not skip any elements
   windowStep = math.min(windowStep, windowSize)
   
   -- Compute indexes of elements in each window
   local indexes = torch.range(1, elemsNum)
   local windowIndexes = indexes:unfold(1, windowSize, windowStep)
   
   local nWins = windowIndexes:size(1)
   
   -- Use this for a virtually non-overlapping sliding window, i.e.
   -- make each element appears in only one window, but the window
   -- start and end pointers still follow given window step and size
   -- (The last step then uses a smaller window)
   local skipLength = 0
   if no_overlap then
      skipLength = windowSize - windowStep
   end
   
   local function get_window(win_n)
      local currentIndexes
      
      if win_n > nWins then
	 if windowIndexes[-1][-1] < elemsNum then
	    -- Make a last window with all elements not yet returned
	    if no_overlap then
	       local lastIndex = windowIndexes[-1][-1]
	       -- Non overlapping last window
	       currentIndexes = torch.range(lastIndex+1, elemsNum)
	    else
	       -- Potentially overlaping last-window
	       currentIndexes = torch.range(elemsNum - (windowSize-1), elemsNum)
	    end
	 else
	    return {}
	 end
      else
	 -- Normal window
	 currentIndexes = windowIndexes[win_n]  -- Indexes for current window
      end
      
      local window = {}
      
      local content_start = 1
      if win_n > 1 and win_n <= nWins then
	 -- Take non-overlapping sliding window in account
	 content_start = 1 + skipLength
      end
      
      for i=content_start, currentIndexes:size(1) do
	 window[i-content_start + 1] = elems[currentIndexes[i]]
      end
      
      local win_start = currentIndexes[1]
      local win_end = currentIndexes[-1]
      return window, win_start, win_end
   end
   
   local win_n = 1
   
   return function()
      collectgarbage(); collectgarbage()
      if win_n > nWins+1 then return nil end
      
      -- get new files to load
      local window, cur_start, cur_end = get_window(win_n)
      
      -- if window is empty
      if next(window) == nil then return nil end

      local output = f(window, cur_start, cur_end)

      win_n = win_n + 1
      
      return output, cur_start, cur_end
   end
end

return M
