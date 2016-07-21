-- Sliding window iterator on set of elements

local M = {}

--
function M.foreach_windowed(f, elems, windowSize, windowStep, no_overlap, verbose)
   local elems_num = #elems
   local windowSize = math.min(windowSize, elems_num)
   local windowStep = math.max(windowStep or math.floor(windowSize / 10), 1)

   -- Use this for a non-overlapping sliding window (last step then uses
   -- a smaller window)
   local no_overlap = no_overlap or false
   
   -- Avoid skipping some examples because of a step too long
   if windowStep > windowSize then
      windowStep = windowSize
   end
   
   local function euclidean_division(a, b)
      local quotient = math.floor(a / b)
      local remainder = a % b
      
      return quotient, remainder
   end
   
   local steps_num, remainder_step = euclidean_division(
      elems_num - windowSize, windowStep)
   if not(steps_num >= 0) then
      steps_num = 0
   end
   
   local sizes = {}
   sizes['winSize'] = windowSize
   sizes['stepSize'] = windowStep
   sizes['lastStep'] = remainder_step
   
   if verbose then
      print('Sizes:')
      print(sizes)
      print('Main steps number: ' .. steps_num)
   end
   
   -- Current position of window on elements
   local cur_start = 1
   local cur_end = sizes['winSize']
   
   local function get_window(step_n)
      if verbose then
	 print('Step_n: ' .. step_n)
      end

      if step_n > steps_num + 1 or (step_n > 0 and cur_end >= elems_num) then
	 return {}, cur_end, cur_end
      end
      
      local window = {}
      
      local win_size = sizes['winSize']
      local step_size = sizes['stepSize']
      
      -- get length of next step
      local next_step
      local is_last_step = false
      if step_n <= steps_num then
	 next_step = step_size  -- a normal step
      else
	 if verbose then print('Last step!') end
	 is_last_step = true
	 -- in that case, step_n == steps_num, make smaller step
	 -- with the remaining files
	 if no_overlap then
	    next_step = sizes['winSize']
	    win_size = sizes['lastStep']
	 else
	    next_step = sizes['lastStep']
	 end
      end
      
      local win_start = (step_n-1) * step_size + next_step + 1
      local win_end = win_start + win_size - 1
      
      local content_win_start = win_start
      if is_last_step and no_overlap then win_start = win_end - windowSize + 1 end

      if verbose then
	 print('Window start: ' .. win_start)
      end
      
      if no_overlap and step_n > 0 then
      	 -- Only return new elements within the sliding window
      	 -- (i.e. end of full window)
	 if not is_last_step then
	    content_win_start = win_start + (win_size - next_step)
	 end
      end
      
      if verbose then
	 if no_overlap then
	    print('Window start (trimmed for new content): ' .. content_win_start)
	 end
	 print('Window end: ' .. win_end)
	 print('Remaning elements: ' .. elems_num-win_end)
      end
      
      if verbose and win_end == elems_num then
	 print('Reached end of input elements')
      end
      
      for elem_n = content_win_start, win_end do
	 window[1 + elem_n-content_win_start] = elems[elem_n]
      end
      
      return window, win_start, win_end
   end

   local step_n = 0
   
   return function()
      collectgarbage(); collectgarbage()
      if step_n > steps_num+1 then return nil end
      
      -- get new files to load
      local window, cur_start, cur_end = get_window(step_n)
      
      -- if window is empty
      if next(window) == nil then return nil end

      -- slice all examples in the batch to have same duration,
      -- allows putting them all in a single tensor for memory efficiency
      local output = f(window, cur_start, cur_end)

      step_n = step_n + 1
      
      return output, cur_start, cur_end
   end
end

return M
