-- https://github.com/lisa-lab/pylearn2/blob/14b2f8bebce7cc938cfa93e640008128e05945c1/pylearn2/datasets/preprocessing.py

function zca_whiten_fit(data, bias)
   local bias= bias or 1e-1
   local auxdata = data:clone()
   local dims = data:size()
   local nsamples = dims[1]
   local n_dimensions = data:nElement() / nsamples
   if data:dim() >= 3 then
      auxdata = auxdata:view(nsamples, n_dimensions)
   end
   -- Center data
   means = torch.mean(auxdata, 1):squeeze()
   auxdata = auxdata - torch.ger(torch.ones(nsamples),means)
   bias = torch.eye(n_dimensions)*bias
   c = torch.mm(auxdata:t(),auxdata)
   c:div(nsamples):add(bias)
   local ce,cv = torch.symeig(c,'V')
   ce:sqrt()
   local invce = ce:clone():pow(-1)
   local invdiag = torch.diag(invce)
   P = torch.mm(cv, invdiag)
   P = torch.mm(P, cv:t())
   return means, P  --, invP
end

function zca_whiten_apply(data, means, P)
   local auxdata = data:clone()
   local dims = data:size()
   local nsamples = dims[1]
   local n_dimensions = data:nElement() / nsamples
    if data:dim() >= 3 then
       auxdata = auxdata:view(nsamples, n_dimensions)
    end
   local xmeans = means:new():view(1,n_dimensions):expand(nsamples,n_dimensions)
   auxdata:add(-1, xmeans)
   auxdata = torch.mm(auxdata, P)
   auxdata:resizeAs(data)
   return auxdata
end

function gcn(x, scale, bias, epsilon)
   local scale = scale or 55
   local bias = bias or 0
   local epsilon = epsilon or 1e-8
   if x:dim() > 2 then
      local num_samples = x:size(1)
      local length = x:nElement()/num_samples
      x = x:reshape(num_samples, length)
   elseif x:dim() < 2 then
      assert(false)
   end
   -- subtract mean: x = x - mean(x)
   local m = torch.ger(x:mean(2):squeeze(), torch.ones(x:size(2)))
   local xm = torch.add(x, -1, m)
   -- calculate normalizer
   local x_std_v = torch.pow(xm, 2):sum(2):add(bias):sqrt():div(scale)
   x_std_v[torch.lt(x_std_v, epsilon)]:fill(1)
   -- divide by normalizer
   local x_std = torch.ger(x_std_v:mean(2):squeeze(), torch.ones(x:size(2)))
   local x_norm = torch.cdiv(xm, x_std)
   return x_norm
end
