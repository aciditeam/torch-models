------------------------------------------------------------------------
-- [[ hyperSampler ]]
-- hyper parameter sampling distributions 
------------------------------------------------------------------------

local hyperSampler = torch.class("hyperSampler")

-- sample from a categorical distribution
function hyperSampler:categorical(probs, vals)
   assert(torch.type(probs) == 'table', "Expecting table of probabilites, got :"..tostring(probs))
   local probs = torch.Tensor(probs)
   local idx = torch.multinomial(probs, 1)[1]
   local val = vals and vals[idx] or idx
   return val
end

-- sample from a normal distribution
function hyperSampler:normal(mean, std)
   assert(torch.type(mean) == 'number')
   assert(torch.type(std) == 'number')
   local val = torch.normal(mean, std)
   return val
end

-- sample from uniform distribution
function hyperSampler:uniform(minval, maxval)
   assert(torch.type(minval) == 'number')
   assert(torch.type(maxval) == 'number')
   local val = torch.uniform(minval, maxval)
   return val
end

-- Returns a value drawn according to exp(uniform(low, high)) 
-- so that the logarithm of the return value is uniformly distributed.
-- When optimizing, this variable is constrained to the interval [exp(low), exp(high)].
function hyperSampler:logUniform(minval, maxval)
   assert(torch.type(minval) == 'number')
   assert(torch.type(maxval) == 'number')
   local val = torch.exp(torch.uniform(minval, maxval))
   return val
end

-- sample from uniform integer distribution
function hyperSampler:randint(minval, maxval)
   assert(torch.type(minval) == 'number')
   assert(torch.type(maxval) == 'number')
   local val = torch.random(minval, maxval)
   return val
end
