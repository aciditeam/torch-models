require 'torch'
require 'nn'

local VAE = {}

function VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
     -- The Encoder
    local encoder = nn.Sequential()
    encoder:add(nn.Linear(input_size, hidden_layer_size))
    encoder:add(nn.ReLU(true))
    
    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))
    mean_logvar:add(nn.Linear(hidden_layer_size, latent_variable_size))

    encoder:add(mean_logvar)
    
    return encoder
end

function VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)
    -- The Decoder
    local decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, hidden_layer_size))
    decoder:add(nn.ReLU(true))

    if continuous then
        mean_logvar = nn.ConcatTable()
        mean_logvar:add(nn.Linear(hidden_layer_size, input_size))
        mean_logvar:add(nn.Linear(hidden_layer_size, input_size))
        decoder:add(mean_logvar)
    else
        decoder:add(nn.Linear(hidden_layer_size, input_size))
        decoder:add(nn.Sigmoid(true))
    end

    return decoder
end

--[[

-- Joost van Amersfoort - <joost@joo.st>
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

nngraph.setDebug(false)

local VAE = require 'VAE'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'

--For loading data files
require 'load'

local continuous = false
data = load(continuous)

local input_size = data.train:size(2)
local latent_variable_size = 20
local hidden_layer_size = 400

local batch_size = 100

torch.manualSeed(1)

local encoder = VAE.get_encoder(input_size, hidden_layer_size, latent_variable_size)
local decoder = VAE.get_decoder(input_size, hidden_layer_size, latent_variable_size, continuous)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})

local reconstruction, reconstruction_var, model
if continuous then
    reconstruction, reconstruction_var = decoder(z):split(2)
    model = nn.gModule({input},{reconstruction, reconstruction_var, mean, log_var})
    criterion = nn.GaussianCriterion()
else
    reconstruction = decoder(z)
    model = nn.gModule({input},{reconstruction, mean, log_var})
    criterion = nn.BCECriterion()
    criterion.sizeAverage = false
end

-- Some code to draw computational graph
-- dummy_x = torch.rand(dim_input)
-- model:forward({dummy_x})

-- Uncomment to get structure of the Variational Autoencoder
-- graph.dot(.fg, 'Variational Autoencoder', 'VA')

KLD = nn.KLDCriterion()

local parameters, gradients = model:getParameters()

local config = {
    learningRate = 0.001
}

local state = {}

epoch = 0
while true do
    epoch = epoch + 1
    local lowerbound = 0
    local tic = torch.tic()

    local shuffle = torch.randperm(data.train:size(1))

    -- This batch creation is inspired by szagoruyko CIFAR example.
    local indices = torch.randperm(data.train:size(1)):long():split(batch_size)
    indices[#indices] = nil
    local N = #indices * batch_size

    local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        local inputs = data.train:index(1,v)

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            model:zeroGradParameters()
            local reconstruction, reconstruction_var, mean, log_var
            if continuous then
                reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(inputs))
                reconstruction = {reconstruction, reconstruction_var}
            else
                reconstruction, mean, log_var = unpack(model:forward(inputs))
            end

            local err = criterion:forward(reconstruction, inputs)
            local df_dw = criterion:backward(reconstruction, inputs)

            local KLDerr = KLD:forward(mean, log_var)
            local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))

            if continuous then
                error_grads = {df_dw[1], df_dw[2], dKLD_dmu, dKLD_dlog_var}
            else
                error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
            end

            model:backward(inputs, error_grads)

            local batchlowerbound = err + KLDerr

            return batchlowerbound, gradients
        end

        x, batchlowerbound = optim.adam(opfunc, parameters, config, state)

        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("Epoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. torch.toc(tic)) 

    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
    end

    if epoch % 2 == 0 then
        torch.save('save/parameters.t7', parameters)
        torch.save('save/state.t7', state)
        torch.save('save/lowerbound.t7', torch.Tensor(lowerboundlist))
    end
end

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(mean, log_var)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    local mean_sq = torch.pow(mean, 2)
    local KLDelements = log_var:clone()

    KLDelements:exp():mul(-1)
    KLDelements:add(-1, mean_sq)
    KLDelements:add(1)
    KLDelements:add(log_var)

    self.output = -0.5 * torch.sum(KLDelements)

    return self.output
end

function KLDCriterion:updateGradInput(mean, log_var)
  self.gradInput = {}

    self.gradInput[1] = mean:clone()

    -- Fix this to be nicer
    self.gradInput[2] = torch.exp(log_var):mul(-1):add(1):mul(-0.5)

    return self.gradInput
end

require 'nn'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:updateOutput(input, target)
    -- - log(sigma) - 0.5 *(2pi)) - 0.5 * (x - mu)^2/sigma^2
    -- input[1] = mu
    -- input[2] = log(sigma^2)

    local Gelement = torch.mul(input[2],0.5):add(0.5 * math.log(2 * math.pi))
    Gelement:add(torch.add(target,-1,input[1]):pow(2):cdiv(torch.exp(input[2])):mul(0.5))

    self.output = torch.sum(Gelement)

    return self.output
end

function GaussianCriterion:updateGradInput(input, target)
    self.gradInput = {}

    -- (x - mu) / sigma^2  --> (1 / sigma^2 = exp(-log(sigma^2)) )
    self.gradInput[1] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1])):mul(-1)

    -- - 0.5 + 0.5 * (x - mu)^2 / sigma^2
    self.gradInput[2] = torch.exp(-input[2]):cmul(torch.add(target,-1,input[1]):pow(2)):mul(-1):add(0.5)

    return self.gradInput
end

--]]