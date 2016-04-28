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
require 'torch'
local uuid = require 'uuid'
local json = require 'cjson'
Plot = require 'itorch.Plot'

visualMonitor = torch.class('visualMonitor');

function visualMonitor:__init(file, type)
  self.file = file or 'monitor.html'
  self.type = type or 'full'
  self.monitorModules = {}        -- Modules that we are monitoring
  self.state = {'idle'};          -- States of monitoring (idle, pretrain, train, finetuning)
  self.plotTables = {}            -- Check the learning state
end

function visualMonitor:refresh()
  self:render();
  -- Here need to send refresh request
  os.execute('refresh_state ' .. pid .. '')
end

function visualMonitor:registerModule(module, description)
  -- Create a structure to monitor
  monitored = {};
  monitored.module = module;
  monitored.description = description;
  -- Here we simply add the structure to monitoring list
  table.insert(self.monitorModules, monitored)
end

function visualMonitor:state()
end

function visualMonitor:open()
  os.execute('open ' .. file);
end

local function encodeAllModels(m)
  local s = json.encode(m)
  local w = {'selected', 'above', 'geometries', 'right', 'tags'}
  for i=1,#w do
    local before = '"' .. w[i] .. '":{}'
    local after = '"' .. w[i] .. '":[]'
    s=string.gsub(s, before, after)
  end
  return s
end

function visualMonitor:render()
  local gen_divs = '';
  local gen_models = '';
  local modelDefinition = [[if(typeof(Bokeh) !== "undefined") {
  console.log("Bokeh: BokehJS loaded, going straight to plotting");
  
  var modelid = "${model_id}";
  var modeltype = "Plot";
  var all_models = ${all_models};
  Bokeh.load_models(all_models);
  var model = Bokeh.Collections(modeltype).get(modelid);
  $("#${window_id}").html(''); // clear any previous plot in window_id
  var view = new model.default_view({model: model, el: "#${window_id}"});
  
  } else {
  load_lib(bokehjs_url, function() {
      console.log("Bokeh: BokehJS plotting callback run at", new Date())
      
      var modelid = "${model_id}";
      var modeltype = "Plot";
      var all_models = ${all_models};
      Bokeh.load_models(all_models);
      var model = Bokeh.Collections(modeltype).get(modelid);
      $("#${window_id}").html(''); // clear any previous plot in window_id
      var view = new model.default_view({model: model, el: "#${window_id}"});
  
  }); }]]
  local divDefinition = [[<div class="plotdiv" id="${div_id}" style="float:left; width:250px; height:220px; margin-left:0px ; margin-top:0px"></div>]]
  for i = 1,#plotsTable do
  print(i);
  local allmodels = plotsTable[i]:_toAllModels()
  local div_id = uuid.new()
  local window_id = window_id or div_id
  plotsTable[i]._winid = window_id
  -- find model_id
  local model_id
  for k,v in ipairs(allmodels) do
    if v.type == 'Plot' then
      model_id = v.id
      v.attributes.plot_width = 200;
      v.attributes.plot_height = 200;
      v.attributes.logo = 'None'
      v.attributes.toolbar_location = 'None'
      v.attributes.title_font_size = '1pt'
      v.attributes.min_border = 0;
      v.attributes.border_left = 0;
      v.attributes.border_right = 0;
      v.attributes.border_up = 0;
      v.attributes.border_down = 0;
    end
  end
  assert(model_id, "Could not find Plot element in input allmodels");
  local html = modelDefinition % {
    window_id = window_id,
    div_id = div_id,
    all_models = encodeAllModels(allmodels),
    model_id = model_id
  };
  local div = divDefinition % {
    div_id = div_id
  };
  gen_models = gen_models .. html
  gen_divs = gen_divs .. div
  end
  local base_template = [[ <script type="text/javascript">
  $(function() {
    if (typeof (window._bokeh_onload_callbacks) === "undefined"){ window._bokeh_onload_callbacks = []; }
    function load_lib(url, callback) { window._bokeh_onload_callbacks.push(callback);
    if (window._bokeh_is_loading){ console.log("Bokeh: BokehJS is being loaded, scheduling callback at", new Date()); return null; }
    console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", new Date());
    window._bokeh_is_loading = true;
    var s = document.createElement('script'); s.src = url; s.async = true;
    s.onreadystatechange = s.onload = function(){
      Bokeh.embed.inject_css("https://cdn.pydata.org/bokeh-0.7.0.min.css");
      window._bokeh_onload_callbacks.forEach(function(callback){callback()});
    };
    s.onerror = function() { console.warn("failed to load library " + url); };
    document.getElementsByTagName("head")[0].appendChild(s);
    }
    bokehjs_url = "https://cdn.pydata.org/bokeh-0.7.0.min.js"
    ]] .. gen_models .. [[
    });
    </script>
  ]]

  local html = 
  [[ <!DOCTYPE html>
  <html lang="en">
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://cdn.pydata.org/bokeh-0.7.0.min.css" type="text/css" />
        <script type="text/javascript" src="https://cdn.pydata.org/bokeh-0.7.0.js"></script> ]] 
    .. base_template .. 
    [[ </head>
    <body> ]] 
    .. gen_divs .. 
    [[ </body>
  </html> ]]
  return html
end

function plotTimeSeries(y)
  print(y);
  plot = Plot();
  -- Generate an index vector
  x = torch.linspace(1, y:size(1), y:size(1));
  plot:line(x, y, 'red', 'time series')
  plot:legend(true)
  plot:title('Line Plot Demo')
  plot:draw()
  plot:save('fig.html');
  os.execute('open fig.html');
end

function generateHTMLColor()
  htmlCodes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
  color = '#'
  for i = 1,6 do
    color = color .. htmlCodes[math.random(#htmlCodes)];
  end
  return color;
end

function multiPlot(plotsTable)
  local gen_divs = '';
  local gen_models = '';
  local modelDefinition = [[if(typeof(Bokeh) !== "undefined") {
  console.log("Bokeh: BokehJS loaded, going straight to plotting");
  var modelid = "${model_id}";
  var modeltype = "Plot";
  var all_models = ${all_models};
  Bokeh.load_models(all_models);
  var model = Bokeh.Collections(modeltype).get(modelid);
  $("#${window_id}").html(''); // clear any previous plot in window_id
  var view = new model.default_view({model: model, el: "#${window_id}"});
    } else {
  load_lib(bokehjs_url, function() {
      console.log("Bokeh: BokehJS plotting callback run at", new Date())
      var modelid = "${model_id}";
      var modeltype = "Plot";
      var all_models = ${all_models};
      Bokeh.load_models(all_models);
      var model = Bokeh.Collections(modeltype).get(modelid);
      $("#${window_id}").html(''); // clear any previous plot in window_id
      var view = new model.default_view({model: model, el: "#${window_id}"});
  }); }]]
  local divDefinition = [[<div class="plotdiv" id="${div_id}" style="float:left; width:250px; height:220px; margin-left:0px ; margin-top:0px"></div>]]
  for i = 1,#plotsTable do
  print(i);
  local allmodels = plotsTable[i]:_toAllModels()
  local div_id = uuid.new()
  local window_id = window_id or div_id
  plotsTable[i]._winid = window_id
  -- find model_id
  local model_id
  for k,v in ipairs(allmodels) do
    if v.type == 'Plot' then
      model_id = v.id
      v.attributes.plot_width = 200;
      v.attributes.plot_height = 200;
      v.attributes.logo = 'None'
      v.attributes.toolbar_location = 'None'
      v.attributes.title_font_size = '1pt'
      v.attributes.min_border = 0;
      v.attributes.border_left = 0;
      v.attributes.border_right = 0;
      v.attributes.border_up = 0;
      v.attributes.border_down = 0;
    end
  end
  assert(model_id, "Could not find Plot element in input allmodels");
  local html = modelDefinition % {
    window_id = window_id,
    div_id = div_id,
    all_models = encodeAllModels(allmodels),
    model_id = model_id
  };
  local div = divDefinition % {
    div_id = div_id
  };
  gen_models = gen_models .. html
  gen_divs = gen_divs .. div
  end
  local base_template = [[ <script type="text/javascript">
  $(function() {
    if (typeof (window._bokeh_onload_callbacks) === "undefined"){ window._bokeh_onload_callbacks = []; }
    function load_lib(url, callback) { window._bokeh_onload_callbacks.push(callback);
    if (window._bokeh_is_loading){ console.log("Bokeh: BokehJS is being loaded, scheduling callback at", new Date()); return null; }
    console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", new Date());
    window._bokeh_is_loading = true;
    var s = document.createElement('script'); s.src = url; s.async = true;
    s.onreadystatechange = s.onload = function(){
      Bokeh.embed.inject_css("https://cdn.pydata.org/bokeh-0.7.0.min.css");
      window._bokeh_onload_callbacks.forEach(function(callback){callback()});
    };
    s.onerror = function() { console.warn("failed to load library " + url); };
    document.getElementsByTagName("head")[0].appendChild(s);
    }
    bokehjs_url = "https://cdn.pydata.org/bokeh-0.7.0.min.js"
    ]] .. gen_models .. [[
    });
    </script>
  ]]

  local html = 
  [[ <!DOCTYPE html>
  <html lang="en">
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://cdn.pydata.org/bokeh-0.7.0.min.css" type="text/css" />
        <script type="text/javascript" src="https://cdn.pydata.org/bokeh-0.7.0.js"></script> ]] 
    .. base_template .. 
    [[ </head>
    <body> ]] 
    .. gen_divs .. 
    [[ </body>
  </html> ]]
  return html
end

function plotMultivariateTimeSeries(y)
  plot = Plot()
  -- Two modes
  if torch.type(y) == 'table' then dimSeries = 1; nSeries = #y; else dimSeries = 1; nSeries = y:size(1) end
  print(#y);
  print(nSeries)
  -- Generate index vector
  x = torch.linspace(1, y[1]:size(dimSeries), y[1]:size(dimSeries));  
  for i = 1,nSeries do
    curTS = y[i];
    plot:line(x, curTS, generateHTMLColor(), 'Series' .. i)
  end
  plot:legend(true)
  plot:title('Line Plot Demo')
  plot:draw()
  plot:save('fig.html');
  os.execute('open fig.html');
end

function plotMultipleTimeSeries(y)
  -- Two modes
  if torch.type(y) == 'table' then dimSeries = 1; nSeries = #y; else dimSeries = 1; nSeries = y:size(1) end
  print(#y);
  print(nSeries)
  -- Generate index vector
  plotTable = {}; 
  for i = 1,nSeries do
    local plot = Plot()
    curTS = y[i];
    x = torch.linspace(1, curTS:size(dimSeries), curTS:size(dimSeries));
    plot:line(x, curTS, generateHTMLColor())
    plot:title('');
    plot:draw()
    plotTable[i] = plot;
  end
  html = multiPlot(plotTable);
  local f = assert(io.open('fig.html', 'w'),
                'filename cannot be opened in write mode')
  f:write(html)
  f:close()
  os.execute('open fig.html')
end
  
-- scatter plots
function plotScatter(x, y, groups, groupNames)
  plot = Plot();
  for i = 1,#groups do
    plot:circle(x[groups[i]], y[groups[i]], 'red', groupNames[i]);
  end
  plot:title('Scatter Plot Demo'):redraw()
  plot:xaxis('length'):yaxis('width'):redraw()
  plot:legend(true)
  plot:redraw()
  plot:save('fig.html')
  os.execute('open fig.html')
end



----------------------------------------------------------------------
-- Export the weights to image
----------------------------------------------------------------------
function exportWeights(model, baseFile)
  linearModules = model:findModules('nn.Linear')
  for i = 1, #conv_nodes do
    image.save(baseFile .. 'weights_' .. i, linearModules[i].weight);
  end
end