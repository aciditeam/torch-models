----------------------------------------------------------------------
--
-- Deep time series learning: Analysis of Torch
--
-- UCR Dataset specific functions
--
----------------------------------------------------------------------

----------------------------------------------------------------------
-- Datasets variables
----------------------------------------------------------------------

-- Change this directory to point on all UCR datasets
local baseDir = '/home/aciditeam/datasets/TS_Datasets';

local setList = {'50words','Adiac','ArrowHead','ARSim','Beef',
	   'BeetleFly','BirdChicken','Car','CBF','Chlorine','CinECG',
	   'Coffee','Computers','Cricket_X','Cricket_Y','Cricket_Z','DiatomSize',
	   'DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect',
	   'DistalPhalanxTW','Earthquakes','ECG200','ECG5000','ECGFiveDays',
	   'ElectricDevices','FaceAll','FaceFour','FacesUCR','Fish','FordA','FordB',
	   'Gun_Point','Ham','HandOutlines','Haptics','Herring','InlineSkate',
	   'InsectWingbeatSound','Ionosphere','ItalyPower','LargeKitchenAppliances',
	   'Lighting2','Lighting7','MALLAT','Meat','MedicalImages',
	   'MiddlePhalanxOutlineAgeGroup','MiddlePhalanxOutlineCorrect',
	   'MiddlePhalanxTW','MoteStrain','NonInv_ECG1','NonInv_ECG2','OliveOil',
	   'OSULeaf','PhalangesOutlinesCorrect','Phoneme','Plane',
	   'ProximalPhalanxOutlineAgeGroup','ProximalPhalanxOutlineCorrect',
	   'ProximalPhalanxTW','RefrigerationDevices','ScreenType','ShapeletSim',
	   'ShapesAll','SmallKitchenAppliances','Sonar','SonyAIBO1','SonyAIBO2',
	   'StarLight','Strawberry','SwedishLeaf','Symbols','Synthetic',
	   'ToeSegmentation1','ToeSegmentation2','Trace','Two_Patterns','TwoLeadECG',
	   'uWGestureX','uWGestureY','uWGestureZ','Vehicle','Vowel','Wafer','Waveform','Wdbc','Wine',
	   'Wins','WordsSynonyms','Worms','WormsTwoClass','Yeast','Yoga'};

--setList = {'ArrowHead','Beef','BeetleFly','BirdChicken'};
--setList = {'50words','Adiac','ArrowHead','ARSim','Beef','BeetleFly','BirdChicken'};

----------------------------------------------------------------------
-- Datasets parsing and processing
----------------------------------------------------------------------

function parse(fileName)
   -- Load the ASCII tab-separated file
   local csvFile = io.open(fileName, 'r');
   -- Prepare a data table 
   local data = {};
   -- Class indexes
   local classes = {};
   local i = 1;
   -- Parse lines of file
   for line in csvFile:lines('*l') do
      data[i] = {};
      j = 1;
      for val in string.gmatch(line, "%S+") do
	 if (j == 1) then
            classes[i] = tonumber(val);
	 else
            data[i][j-1] = tonumber(val);
	 end
	 j = j + 1;
      end
      i = i + 1;
   end
   csvFile:close();
   local finalData = torch.Tensor(data);
   return finalData
end
