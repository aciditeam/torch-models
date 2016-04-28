------------------------------------------------------------------
-- istaencoder class (possibly with regularizer on code)

require 'nn'

------------------------------------------------------------------
local istastep, parent = torch.class('nn.istastep', 'nn.Module')

function istastep:__init(outputsize)
    parent.__init(self)
    self.S = nn.Linear(outputsize, outputsize)
    self.G = nn.SoftShrink()
    self.hidden = nil
    self.gradCode = nil
    self.gradZ = nil
end

function istastep:updateOutput(z, code)
    self.hidden = torch.add(code, self.S:updateOutput(z))
    self.output = self.G:updateOutput(self.hidden)
    return self.output
end

function istastep:updateGradInput(z, code, gradOutput)
    self.gradCode = self.G:updateGradInput(self.hidden, gradOutput)
    self.gradZ = self.S:updateGradInput(z, self.gradCode)
    self.gradInput = self.gradZ, self.gradCode
   return self.gradZ, self.gradCode
end

function istastep:accGradParameters(z, code, gradOutput)
    self.G:accGradParameters(self.hidden, gradOutput)
    self.S:accGradParameters(z, self.gradCode)
end

function istastep:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i = 1, #from do tinsert(to, from[i]) end
        else
            table.insert(to, from)
        end
    end
    local w = {}
    local gw = {}
    local mw, mgw = self.S:parameters()
    if mw then tinsert(w, mw) tinsert(gw, mgw) end
    local mw, mgw = self.G:parameters()
    if mw then tinsert(w, mw) tinsert(gw, mgw) end
    return w, gw
end

------------------------------------------------------------------
local istaencoder, parent = torch.class('nn.istaencoder', 'nn.Module')

function istaencoder:__init(inputsize, outputsize, n)
    parent.__init(self)
    self.encoder = nn.Linear(inputsize, outputsize)
    self.G0 = nn.SoftShrink()
    -- array of ista step modules
    self.istasteps = {}
    self.code0 = nil
    self.gradCode0 = nil
    for i = 1, n do
        self.istasteps[i] = nn.istastep(outputsize)
    end
end

function istaencoder:updateOutput(input)
    -- print(input:size())
    -- print('updating output')
    self.code0 = self.encoder:updateOutput(input)
    local z = self.G0:updateOutput(self.code0)
    for i = 1, #self.istasteps do
        z = self.istasteps[i]:updateOutput(z, self.code0)
    end
    self.output = z
    return self.output
end

function istaencoder:updateGradInput(input, gradOutput)
    -- print('updating grad input')
    local gradZ = gradOutput
    local gradTemp = nil
    if self.gradCode0 == nil then self.gradCode0 = torch.Tensor(gradOutput:size()) end
    self.gradCode0:zero()
    if #self.istasteps > 1 then
        for i = #self.istasteps, 2, -1 do
            gradZ, gradTemp = self.istasteps[i]:updateGradInput(self.istasteps[i - 1].output, self.code0, gradZ)
            self.gradCode0 = torch.add(self.gradCode0, gradTemp)
        end
        gradZ, gradTemp = self.istasteps[1]:updateGradInput(self.G0.output, self.code0, gradZ)
        self.gradCode0 = torch.add(self.gradCode0, gradTemp)
    end
    gradTemp = self.G0:updateGradInput(self.code0, gradZ)
    self.gradCode0 = torch.add(self.gradCode0, gradTemp)
    self.gradInput = self.encoder:updateGradInput(input, self.gradCode0)
    return self.gradInput
end

function istaencoder:accGradParameters(input, gradOutput)
    local gradZ = gradOutput
    local gradTemp = nil
    if #self.istasteps > 1 then
        for i = #self.istasteps, 2, -1 do
            self.istasteps[i]:accGradParameters(self.istasteps[i - 1].output, self.code0, gradZ)
            gradZ = self.istasteps[i].gradInput
        end
        self.istasteps[1]:accGradParameters(self.G0.output, self.code0, gradZ)
        gradZ = self.istasteps[1].gradInput
    end
    self.G0:accGradParameters(self.code0, gradZ)
    self.encoder:accGradParameters(input, self.gradCode0)
end

-- collect the parameters so they can be flattened
-- this assumes that the cost doesn't have parameters.
function istaencoder:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i = 1, #from do tinsert(to, from[i]) end
        else
            table.insert(to, from)
        end
    end
    local w = {}
    local gw = {}
    local mw, mgw = self.encoder:parameters()
    if mw then tinsert(w, mw) tinsert(gw, mgw) end
    local mw, mgw = self.G0:parameters()
    if mw then tinsert(w, mw) tinsert(gw, mgw) end
    for i = 1, #self.istasteps do
        local mw, mgw = self.istasteps[i]:parameters()
        if mw then tinsert(w, mw) tinsert(gw, mgw) end
    end
    return w, gw
end


function istaencoder:weights()
    return self.encoder.weight
end

------------------------------------------------------------------
-- an auto-encoder with a regularizer on the code vector
local registaencoder, parent = 
    torch.class('nn.registaencoder', 'nn.istaencoder')

function registaencoder:__init(encoder, decoder, cost, regularizer)
    parent.__init(self)
    self.encoder = encoder
    self.decoder = decoder
    self.cost = cost
    self.regularizer = regularizer   -- regularizer module
    self.code = 0
    self.gradcode = 0
    self.recons = 0
    self.gradrecons = 0
    self.alpha = 1.0         -- coefficient of regularizer
    self.recenergy = 0       -- reconstruction energy
    self.regenergy = 0       -- regularizer energy
    self.energy = 0          -- total energy (sum of the above)
end

function registaencoder:updateOutput(input)
    self.code = self.encoder:updateOutput(input)
    self.regenergy = self.regularizer:updateOutput(self.code)
    self.recons = self.decoder:updateOutput(self.code)
    self.recenergy = self.cost:updateOutput(self.recons, input)
    self.energy = self.regenergy + self.recenergy
    return self.energy
end

function registaencoder:updateGradInput(input)
    self.gradrecons = self.cost:updateGradInput(self.recons, input)
    self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons) +
                    self.regularizer:updateGradInput(self.code)
    return self.encoder:updateGradInput(input, self.gradcode)
end

function registaencoder:accGradParameters(input)
    self.gradrecons = self.cost:updateGradInput(self.recons, input)
    -- self.gradrecons = self.cost:accGradParameters(recons, input)
    self.gradcode = self.decoder:updateGradInput(self.code, self.gradrecons) +
                    self.regularizer:updateGradInput(self.code)
    self.decoder:accGradParameters(self.code, self.gradrecons)
    self.encoder:updateGradInput(input, self.gradcode)
    self.encoder:accGradParameters(input, self.gradcode)
end


