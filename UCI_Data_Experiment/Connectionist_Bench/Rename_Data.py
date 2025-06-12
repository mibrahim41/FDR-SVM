import glob
import os

all_files = glob.glob('FDR-SVM_CB_*.mat')
count = 1
for file in all_files:
    new_name = 'FDR-SVM_CB_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1
    
