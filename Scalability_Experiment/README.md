The code contained within this directory is used to generate the results of the Scalability Experiment in Appendix C.5

In order to run this experiment please follow the instructions below. 
Please note that while this experiment involves data generation, this is done automatically within the Python files.

1)  Run each of the provided python files starting with 'FDR-SVM...' the desired number of repetitions. 
    Each run should save a .mat file to the current directory. Note that each file tests a different setting.
    'G': increasing clients
    'N': increasing training samples
    'P': increasing features
    'NG': increasing clients and training samples

2)  Run the 'Rename_Files.py' code, which is responsible for renaming all the .mat files sequentially.

3)  Run the 'Scalability_Plotting.m' file, which generates the plots present in the paper.