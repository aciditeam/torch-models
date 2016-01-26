

-- create Gabor with center frequency w0 and variance sigma
function gabor(N, sigma, w0)
	local sigma = 1/sigma
	local w = torch.range(0,N-1)
	local f1 = torch.exp(-torch.pow(w/N*2*math.pi - w0,2)/(2*sigma^2))
	local f0 = torch.exp(-torch.pow(w/N*2*math.pi - 2*math.pi - w0,2)/(2*sigma^2))
	local f2 = torch.exp(-torch.pow(w/N*2*math.pi + 2*math.pi - w0,2)/(2*sigma^2))
	--local f1=torch.exp(-torch.pow(w/(2*math.pi*N)-w0,2)/(2*sigma^2))
	--local f0=torch.exp(-torch.pow(w/(2*math.pi*N)-w0+1/(2*math.pi),2)/(2*sigma^2))
	--local f2=torch.exp(-torch.pow(w/(2*math.pi*N)-w0-1/(2*math.pi),2)/(2*sigma^2))
	return f0 + f1 + f2
end

function morlet(N, sigma, w0)
	local g=gabor(N, sigma, w0)
	local g0=gabor(N, sigma, 0)
	return g - g0*g[1]/g0[1]
end



function morletFreq1D(options)
	local sigma0 = 2/math.sqrt(3)
	local J = options.J
	local Q = options.Q
	local P = options.P
	
	-- create logarithmically spaced, bandpass filters
	local xiPsi = torch.Tensor(J+P)
	xiPsi[{{1,J}}] = torch.exp(torch.range(0,1-J,-1)*math.log(2)/Q) * options.xiPsi
	local sigmaPsi = torch.Tensor(J+P+1)
	sigmaPsi[{{1,J}}] = torch.exp(torch.range(0,J-1)*math.log(2)/Q) * options.sigmaPsi

	-- calculate linearly-spaced bandpass filters so that they evenly cover the remaining part of the spectrum
	if options.P > 0 then
		local step = math.pi * math.pow(2,-J/Q) * (1-1/4*sigma0/options.sigmaPhi*math.pow(2,1/Q))/options.P
		xiPsi[{{J+1,J+P}}] = - torch.range(1,options.P)*step + options.xiPsi * math.pow(2,(-J+1)/Q)
		sigmaPsi[{{J+1,J+1+P}}] = options.sigmaPsi * math.pow(2,(J-1)/Q)
	end

	-- calculate the lowpass filter
	local sigmaPhi = options.sigmaPhi * math.pow(2,(J-1)/Q)

	-- convert (spatial) sigmas to (frequential) bandwidths
	local psiBandWidth = torch.cdiv(torch.Tensor(J+P+1):fill(math.pi/2*sigma0),sigmaPsi)
	local phiBandWidth
	if not options.phiDirac then
		phiBandWidth = math.pi/2 * sigma0/sigmaPhi
		--local phiBandWidth = torch.cdiv(math.pi/2*sigma0,sigmaPhi)
	else
		phiBandWidth = 2*math.pi
	end
	return xiPsi, psiBandWidth, phiBandWidth
end



function filterBank(N, options)
	local psiCenter,psiBandWidth, phiBandWidth = morletFreq1D(options)
	local sigma0 = 2/math.sqrt(3)
	local psiSigma = torch.cdiv(torch.Tensor(psiBandWidth:size(1)):fill(sigma0*math.pi/2),psiBandWidth)
	--local phiSigma = torch.cdiv(torch.Tensor(phiBandWidth:size(1)):fill(sigma0*math.pi/2),phiBandWidth)
	local phiSigma = (sigma0*math.pi/2)/phiBandWidth	

	local psi = {}
	for i=1,psiCenter:size(1) do
		psi[i]=morlet(N, psiSigma[i], psiCenter[i])
	end

	local phi = gabor(N, phiSigma, 0)
	return phi, psi
end

-- test
options={}
options.J = 1
options.Q = 8
options.P = 11
options.xiPsi = 3.0112
options.sigmaPsi = 6.9564
options.sigmaPhi = 6.9564
N=131072/2
phi,psi = filterBank(N,options)
