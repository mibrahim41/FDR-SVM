The code contained within this directory is used to generate the results of the UCI Data Experiment in Section 6.1

Please repeat the following instructions for each dataset.

1)  Navigate to the directory of the desired dataset.

2)  Run the Python code associated with the specific experiment, named 'FDR-SVM_{Dataset_Name}.py'. Once a run is complete, it will save a .mat file to the same directory. Run the file for the number of repetitions desired. You should expect a .mat file per repetition.

3)  Run the 'Rename_Data.py' code to rename all the generated .mat files sequentially.

4)  Run the 'obtain_results.m' code. This compiles the results from all the .mat files and creates one tensor per model that contains all the results. Additionally, it outputs a variable named "res_{Model_Name}_ave" containing the mean F-1 score over all repetitions, and one named "res_{Model_Name}_std" containing the standard deviation for each model.

5)  [OPTIONAL] If you would like to run the Wilcoxon signed-rank test to compare the performance of our ADMM algorithm to that of all federated benchmarks, run the file titled "w_test.m" as a final step. The code outputs four variables named "h_{Model_Name}", one per federated benchmark. If the value of this variable for a benchmark is True (logical 1), this indicates that we reject the null hypothesis, and therefore the performance improvement of ADMM is statistically significant.


This concludes the process for running the code for this experiment.
