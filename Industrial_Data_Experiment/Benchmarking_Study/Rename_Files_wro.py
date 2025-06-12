import glob
import os

all_files = glob.glob('FDR-SVM_bench_wro_*.mat')
count = 1
for file in all_files:
    new_name = 'FDR-SVM_bench_wro_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1