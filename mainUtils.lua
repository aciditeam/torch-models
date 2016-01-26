----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Utility functions
--
----------------------------------------------------------------------

function uniqueTable(table)
  local hash = {}
  local res = {}
  for _,v in ipairs(table) do
    if (not hash[v]) then
      res[#res+1] = v -- you could print here instead of saving to result table if you wanted
      hash[v] = true
    end
  end
  return res;
end

function uniqueTensor(table)
  local hash = {}
  local res = {}
  for i = 1,table:size(1) do
    if (not hash[table[i]]) then
      res[#res+1] = table[i] -- you could print here instead of saving to result table if you wanted
      hash[table[i]] = true
    end
  end
  return res;
end