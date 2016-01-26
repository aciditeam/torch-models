----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- Visualization functions
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Imports
require 'image'

----------------------------------------------------------------------
-- Export the weights to image
----------------------------------------------------------------------
function exportWeights(model, baseFile)
  linearModules = model:findModules('nn.Linear')
  for i = 1, #conv_nodes do
    image.save(baseFile .. 'weights_' .. i, linearModules[i].weight);
  end
end