%% Extract and Write Features to Data
ens = simulationEnsembleDatastore(fullfile('.','Data'));

ens.DataVariables = [ens.DataVariables; ...
    "fPeak"; "pLow"; "pMid"; "pHigh"; "pKurtosis"; ...
    "qMean"; "qVar"; "qSkewness"; "qKurtosis"; ...
    "qPeak2Peak"; "qCrest"; "qRMS"; "qMAD"; "qCSRange"];
ens.ConditionVariables = ["LeakFault","BlockingFault","BearingFault"];

while hasdata(ens)

    % Read member data
    data = read(ens);

    % Preprocess and extract features from the member data
    [flow,flowP,flowF,faultValues] = preprocess(data);
    feat = extractCI(flow,flowP,flowF);

    % Add the extracted feature values to the member data
    dataToWrite = [faultValues, feat];
    writeToLastMemberRead(ens,dataToWrite{:})
end

%% Update Data Columns
ens = simulationEnsembleDatastore(fullfile('.','Data'));
reset(ens)
ens.SelectedVariables = [...
    "fPeak","pLow","pMid","pHigh","pKurtosis",...
    "qMean","qVar","qSkewness","qKurtosis",...
    "qPeak2Peak","qCrest","qRMS","qMAD","qCSRange",...
    "LeakFault","BlockingFault","BearingFault"];
idxLastFeature = 14;

% Load the condition indicator data into memory
data = gather(tall(ens));

pdmRecipPump_Parameters %Pump
CAT_Pump_1051_DataFile_imported %CAD

mdl = 'pdmRecipPump';
open_system(mdl)
leak_area_set_factor = [0,1e-3,2e-3,3e-3,4e-3];
leak_area_set = leak_area_set_factor*TRP_Par.Check_Valve.In.Max_Area;
leak_area_set = max(leak_area_set,1e-9); % Leakage area cannot be 0
leak_val_vec = leak_area_set;
data.LeakFlag(data.LeakFault == 1e-9) = 0;
data.LeakFlag(data.LeakFault == leak_val_vec(2)) = 1;
data.LeakFlag(data.LeakFault == leak_val_vec(3)) = 2;
data.LeakFlag(data.LeakFault == leak_val_vec(4)) = 3;
data.LeakFlag(data.LeakFault == leak_val_vec(5)) = 4;

y_vec = zeros(height(data),1);
y_vec(data.LeakFlag == 0) = 0;
y_vec(data.LeakFlag == 1) = 1;
y_vec(data.LeakFlag == 2) = 2;
y_vec(data.LeakFlag == 3) = 3;
y_vec(data.LeakFlag == 4) = 4;

data.BlockingFault = [];
data.BearingFault = [];
data.LeakFault = [];

%% Write Data to Double and Save File
data_arr_x = table2array(data);
data_arr = [data_arr_x,y_vec];