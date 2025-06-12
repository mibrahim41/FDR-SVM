num_exp = 50;
acc_SM_vec = zeros(1,num_exp);
acc_ADMM_vec = zeros(1,num_exp);
acc_ADMM_sc_vec = zeros(1,num_exp);
acc_fedAvg_vec = zeros(1,num_exp);
acc_fedSGD_vec = zeros(1,num_exp);
acc_fedProx_vec = zeros(1,num_exp);
acc_fedDRO_kl_vec = zeros(1,num_exp);
acc_central_vec = zeros(1,num_exp);

for i = 1:num_exp
    filename = sprintf('FDR-SVM_BCW_%d.mat',i);
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

[p_fedSGD, h_fedSGD, stats_fedSGD] = signrank(acc_fedSGD_vec,acc_ADMM_vec,'tail','left');
[p_fedAvg, h_fedAvg, stats_fedAvg] = signrank(acc_fedAvg_vec, acc_ADMM_vec,'tail','left');
[p_fedProx, h_fedProx, stats_fedProx] = signrank(acc_fedProx_vec, acc_ADMM_vec,'tail','left');
[p_fedDRO_kl, h_fedDRO_kl, stats_fedDRO_kl] = signrank(acc_fedDRO_kl_vec, acc_ADMM_vec,'tail','left');