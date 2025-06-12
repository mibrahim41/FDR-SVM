The code contained within this directory is used to generate the results of the Benchmarking Study component of the Industrial Data Experiment in Section 6.2.2

Running this experiment is performed in two stages.
1)  Data generation
2)  Running the models and plotting the results.

Stage 1 - Data Generation:
1)  Navigate to the 'Data_Generation' directory, then to the 'Pump_Code' directory within.

2)  Run the 'FDR_SVM_Script.mat' file, which actually runs the Simulink model and generates the timeseries data.
    The data files will be saved in the 'Data' directory within the active directory. Please do not move the data files for now.

3)  Run the 'Prepare_Data_Script.mat' file, which performs feature extraction on the saved data and updates all the data files
    saved within the 'Data' directory.

4) Save the 'data_arr' variable in the current workspace in a .mat file named "extracted_data_features.mat" in the 'Data' directory.

5)  Navigate to the 'Data' directory.

6)  Run the 'extract_mats.mat' code 5 times, changing the 'client_number' variable each time for each value in {0,1,2,3,4}.
    After each run of this code, save the workspace with the name 'client{client_number}.mat' in the current directory if the client number is 1, 2, 3, or 4.
    If the client number is 0, save the workspace with the name 'nominal.mat' in the current directory.

7)  Copy the saved .mat files to the 'Global' and 'Local' Directory within the root directory of this experiment.


Stage 2 - Running Models (same steps can be repeated for Global and Local):
1)  Navigate to the directory associated with the desired experiment (either 'Global' or 'Local').

2)  Run each of the 5 python codes starting with 'FDR-SVM_sensitivity_{...}.py', each of which tests a different setting in the experiment:
        nom: nominal setting
        gro: client imbalance
        cla: class imbalance
        two: client + class imbalance
        wro: noisy labels
    Make sure to run each file the desired number of reptitions. The code should store a .mat file on the current directory once done.

4)  Run the 'Rename_Data.py' file to rename all the .mat files sequentially.

5)  Run the 'sensitivity_global_plotting.m' file to plot the final results, including error bars.