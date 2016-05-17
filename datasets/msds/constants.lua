-- Million-song Dataset specific constants

local M = {}

-- Training / Validation / Testing splitting for the Million Song Subset

M.subset = {}

M.subset.training = 'A/'

local function compose_suffixes(prefix, suffixes)
   local composed = {}
   for _, suffix in ipairs(suffixes) do
      table.insert(composed, prefix .. suffix)
   end
   return composed
end

-- This validation subset contains 1217 examples
M.subset.validation = compose_suffixes('B/', {'A/', 'B/', 'C/', 'D/'})

-- This testing subset contains 1381 examples
M.subset.testing = compose_suffixes('B/', {'E/', 'F/', 'G/', 'H/', 'I/'})

return M
