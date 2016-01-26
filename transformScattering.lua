require 'image'
require 'fftw'

function scattering(x, filters, delta)
	timer=torch.Timer()
	out={}
	J=filters[1].psi:size(2)
	fin=torch.fft(x)
	out[1]=convSub(fin,complex(filters[1].phi),delta)
	print('1st conv: t=' .. timer:time().real .. 's')
	out[2]=torch.Tensor(math.floor(x:size(1)/delta),J)
	out[3]=torch.Tensor(math.floor(x:size(1)/delta),J,filters[1].decomp[J][1])
	for j=1,J do
		U = torch.fft(complexMod(torch.ifft(complexMul(fin,complex(filters[1].psi[{{},j}])))))
		out[2][{{},j}] = convSub(U, complex(filters[1].phi),delta)
		for j2=1,filters[1].decomp[j][1] do
			local tmp=torch.fft(complexMod(torch.ifft(complexMul(U,complex(filters[2].psi[{{},j2}])))))
			out[3][{{},j,j2}]=convSub(tmp,complex(filters[1].phi),delta)
		end
	end
	print('Time elapsed: ' .. timer:time().real .. ' seconds')
	out[3]:resize(out[3]:size(1),out[3]:size(2)*out[3]:size(3))
	outc=torch.Tensor(out[1]:size(1),1+out[2]:size(2)+out[3]:size(2))
	--outc=torch.Tensor(out[1]:size(1),1+out[2]:size(2)+out[3]:size(2)+out[3]:size(3))		
	outc[{{},1}]=out[1]
	outc[{{},{2,2+out[2]:size(2)-1}}]=out[2]
	outc[{{},{2+out[2]:size(2),2+out[2]:size(2)+out[3]:size(2)-1}}]=out[3]
	return outc
end	


function scattering2(x, filters, delta)
	out = {}
	nLayers = #filters
	n = x:size(1)
	fin = torch.fft(x)
	-- convert filters to complex format
	for i=1,nLayers do
		filters[i].phi:resize(filters[i].phi:nElement())
		filters[i].cpsi = {}
		for j=1,filters[i].psi:size(2) do
			filters[i].cpsi[j]=filters[i].psi[{{},j}]
			filters[i].cpsi[j]:resize(filters[i].cpsi[j]:nElement())
		end
	end
	J = filters[1].psi:size(2)
	out[2] = torch.Tensor(math.floor(n / delta), J)
	out[3] = torch.Tensor(math.floor(n / delta), J, filters[1].decomp[J][1])
	-- apply first lowpass filter
	timer = torch.Timer()
	timer2 = torch.Timer()
	out[1] = lowpass(fin, filters[1].phi, delta)
	print('Time elapsed for lowpass: ' .. timer:time().real .. ' seconds')
	
	-- loop over bands in first layer
	for j = 1,J do
		timer:reset()
		U = bandpass(fin, filters[1].cpsi[j])
		print('Time elapsed for bandpass: ' .. timer:time().real .. ' seconds')
		out[2][{{},j}] = lowpass(U, filters[1].phi, delta)
		local nBands = filters[1].decomp[j][1]
		for k = 1,nBands do
			local tmp = bandpass(U, filters[2].cpsi[k])
			out[3][{{},j,k}] = lowpass(tmp, filters[1].phi, delta)
		end
	end
	print('Total Time elapsed: ' .. timer2:time().real .. ' seconds')
	return out
end



function bandpass(fin, filter, delta)
	return torch.fft(complexMod(torch.ifft(complexMul(fin,filter))))
end

function lowpass(fin, filter, delta)
	return subsample(torch.ifft(complexMul(fin, filter)), delta)
end




function convSub(x, filter, sub)
	local tmp=complexMul(x,filter)
	tmp=torch.ifft(tmp)
	return subsample(tmp,sub)
end


function complex(x)
	local out=torch.zeros(x:nElement(),2)
	out[{{},1}]=x
	return out
end

function complexMod(x)
	if x:nDimension() == 1 then
		return torch.abs(x)
	else
		return torch.norm(x,2)
	end
end

function complexMul(x,y)
	local z=torch.Tensor(x:size())
	if y:nDimension() == 1 then
		z[{{},1}] = torch.cmul(x[{{},1}],y)
		z[{{},2}] = torch.cmul(x[{{},2}],y)
	else
		z[{{},1}] = torch.cmul(x[{{},1}],y[{{},1}]) - torch.cmul(x[{{},2}],y[{{},2}])
		z[{{},2}] = torch.cmul(x[{{},1}],y[{{},2}]) + torch.cmul(x[{{},2}],y[{{},1}])
	end
	return z
end


function subsample(x,d)
	local n=x:nElement()
	x:resize(1,n)
	local out=image.scale(x,math.floor(n/d),1,'simple')
	x:resize(n)
	out:resize(out:nElement())
	return out
end





