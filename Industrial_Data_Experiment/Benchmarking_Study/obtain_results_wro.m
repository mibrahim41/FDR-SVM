num_exp = 10;
acc_SM_vec = zeros(1,num_exp);
acc_ADMM_vec = zeros(1,num_exp);
acc_ADMM_sc_vec = zeros(1,num_exp);
acc_fedAvg_vec = zeros(1,num_exp);
acc_fedSGD_vec = zeros(1,num_exp);
acc_fedProx_vec = zeros(1,num_exp);
acc_fedDRO_kl_vec = zeros(1,num_exp);
acc_central_vec = zeros(1,num_exp);

for i = 1:num_exp
    filename = sprintf('FDR-SVM_bench_wro_%d.mat',i);
    load(filename)
    acc_SM_vec(i) = acc_vec_SM;
    acc_ADMM_vec(i) = acc_vec_ADMM;
    acc_ADMM_sc_vec(i) = acc_vec_ADMM_sc;
    acc_fedAvg_vec(i) = acc_vec_fedAvg;
    acc_fedSGD_vec(i) = acc_vec_fedSGD;
    acc_fedProx_vec(i) = acc_vec_fedProx;
    acc_fedDRO_kl_vec(i) = acc_vec_fedDRO_kl;
    acc_central_vec(i) = acc_vec_central;
end
res_SM_ave = mean(acc_SM_vec);
res_ADMM_ave = mean(acc_ADMM_vec);
res_ADMM_sc_ave = mean(acc_ADMM_sc_vec);
res_fedAvg_ave = mean(acc_fedAvg_vec);
res_fedSGD_ave = mean(acc_fedSGD_vec);
res_fedProx_ave = mean(acc_fedProx_vec);
res_fedDRO_kl_ave = mean(acc_fedDRO_kl_vec);
res_central_ave = mean(acc_central_vec);

res_SM_std = std(acc_SM_vec);
res_ADMM_std = std(acc_ADMM_vec);
res_ADMM_sc_std = std(acc_ADMM_sc_vec);
res_fedAvg_std = std(acc_fedAvg_vec);
res_fedSGD_std = std(acc_fedSGD_vec);
res_fedProx_std = std(acc_fedProx_vec);
res_fedDRO_kl_std = std(acc_fedDRO_kl_vec);
res_central_std = std(acc_central_vec);