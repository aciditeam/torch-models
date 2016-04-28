
-- Solution #1
model = nn.Concat(1);
for l = minGranularity, maxGranularity
model:add(nn.Convolution(l));
model:add(nn.Linear());
model:add(nn.nonLinearity())
end
nn.JoinTable()

-- Solution #1
nn.ParallelTable(l)
for l = minGranularity, maxGranularity
  nn.SlidingWindow(l, 1);
  
  nn.Linear();
end