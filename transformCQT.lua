require 'audio'
require 'image'
require 'fftw'
require 'nn'
require 'cunn'
dofile('complex.lua')
dofile('filters.lua')


cmd = torch.CmdLine()
cmd:option('-chunk', '0', ' dataset chunk (0-9,a-f)')
cmd:option('-winsize',1024, ' size of FFT window')
cmd:option('-stride', 512, ' size of stride between windows')
cmd:option('-octaves', 4, ' number of octaves')
cmd:option('-bands', 24, ' number of bands per octave')
cmd:option('-poolfactor',4,'number of pools per window')
cmd:option('-gpunum',1)
opt = cmd:parse(arg or {})
cutorch.setDevice(opt.gpunum)

dataPath = '/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/' .. opt.chunk .. '/wav/'
savePath = dataPath .. '/CQT_N_' .. opt.winsize .. '_P_' .. opt.poolfactor .. '_O_' .. opt.octaves .. '_R_' .. opt.bands

if paths.dirp(savePath) == false then
	paths.mkdir(savePath)
end

 
-- specify options
sigma0 = 2/math.sqrt(3)
options = {}
options.Q = opt.bands
options.J = opt.bands * opt.octaves
options.P = 0
options.B = options.Q
options.xiPsi = 1/2 * (2^(-1/options.Q)+1) * math.pi
options.sigmaPsi = 1/2 * sigma0/(1-2^(-1/options.B))
options.phiBwMult = 1
options.sigmaPhi = options.sigmaPsi/options.phiBwMult
phi,psi = filterBank(opt.winsize,options)

-- create modules
print('Creating Modules')
nInputPlanes=1
nOutputPlanes=#psi
kW,kH = opt.winsize,1
dW,dH = 1,1

function reverse(x)
	local n = x:size(1)
	local r = torch.Tensor(n)
	for i=1,n do
		r[i] = x[n-i+1]
	end
	return r
end


function circshift(x,shift)
   local n = x:size(1)
   local y = torch.Tensor(n)
   for i=1,n do
      y[i]=x[(i-shift-1) % n + 1]
   end
   return y
end

-- compute real and imaginary temporal kernels 
kernelReal = torch.Tensor(nOutputPlanes,kW)
kernelImag = torch.Tensor(nOutputPlanes,kW)
for i=1,#psi do
	local kT = torch.fft(psi[i])
	kernelReal[{i,{}}]:copy(circshift(reverse(kT[{{},1}]),opt.winsize/2))
	kernelImag[{i,{}}]:copy(circshift(reverse(kT[{{},2}]),opt.winsize/2))
end
kernelReal:resize(nOutputPlanes,1,kW,1)
kernelImag:resize(nOutputPlanes,1,kW,1)

-- put them in convolutional modules
mReal = nn.SpatialConvolution(nInputPlanes,nOutputPlanes,kW,kH,dW,dH)
mImag = nn.SpatialConvolution(nInputPlanes,nOutputPlanes,kW,kH,dW,dH)
mReal.weight:copy(kernelReal)
mImag.weight:copy(kernelImag)
mReal.bias:zero()
mImag.bias:zero()

-- pooling module
pooler = nn.SpatialLPPooling(1,1,opt.winsize/opt.poolfactor,1,opt.winsize/opt.poolfactor)

print('Copying to GPU')
mReal:cuda()
mImag:cuda()
pooler:cuda()

-- dry run to get output size. This way we can preallocate the output and reuse.
file = '/misc/vlgscratch3/LecunGroup/mbhenaff/Magnatagatune/0/wav/rocket_city_riot-last_of_the_pleasure_seekers-08-all_i_got-88-117.mp3.wav'
w = audio.load(file)
w=w:squeeze()
w=w:cuda()
w:resize(1,1,w:size(1))
mReal:forward(w)
mImag:forward(w)
outSize=mReal.output:size()
out = torch.Tensor(1,outSize[1],outSize[3])
out=out:cuda()
timer = torch.Timer()

-- run CQT on all the files in the folder
for file in paths.files(dataPath) do
	if not (file == '.' or file == '..') and string.match(file,'CQT') == nil then
		local saveName = savePath .. '/' .. paths.basename(file,'.mp3.wav') .. '.th'
		if paths.filep(saveName) == true then
			print(saveName .. ' already exists, skipping')
		else
			print('Processing ' .. file)
			timer:reset()
			-- load audio
			local w = audio.load(dataPath .. '/' .. file)
			w=w:squeeze()
			w=w:cuda()
			w:resize(1,1,w:size(1))
			-- convolve with real and imaginary filters
			mReal:forward(w)
			mImag:forward(w)
			-- compute modulus
			out:zero()
			out:cmul(mReal.output,mReal.output)
			out:addcmul(mImag.output,mImag.output)
			out:pow(0.5)
			-- pool
			pooler:forward(out)
			-- save
			local x = pooler.output:float()
			torch.save(saveName,x)
			print('Time=' .. timer:time().real .. ' sec')
		end
	end
end



















