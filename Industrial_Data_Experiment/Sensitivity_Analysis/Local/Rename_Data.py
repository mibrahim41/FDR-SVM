import glob
import os

all_files = glob.glob('FDR-SVM_sensitivity_cla_*.mat')
count = 1
for file in all_files:
    new_name = 'FDR-SVM_sensitivity_cla_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1

all_files = glob.glob('FDR-SVM_sensitivity_wro_*.mat')
count = 1
for file in all_files:
    new_name = 'FDR-SVM_sensitivity_wro_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1

all_files = glob.glob('FDR-SVM_sensitivity_gro_*.mat')
count = 1
for file in all_files:
    new_name = 'FDR-SVM_sensitivity_gro_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1

all_files = glob.glob('FDR-SVM_sensitivity_nom_*.mat')
count = 1
for file in all_files:
    new_name = 'FDR-SVM_sensitivity_nom_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1

all_files = glob.glob('FDR-SVM_sensitivity_two_*.mat')
count = 1
for file in all_files:
    new_name = 'FDR-SVM_sensitivity_two_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1