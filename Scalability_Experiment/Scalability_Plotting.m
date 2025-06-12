G_vec = [10,20,30,40,50];
N_exp = 50;
[accs_G,runtimes_G,adjs_G,errs_G,acc_errs_G] = comp_metrics(G_vec,N_exp,'G');
[accs_G_N,runtimes_G_N,adjs_G_N,errs_G_N,acc_errs_G_N] = comp_metrics(G_vec,N_exp,'G_N');

figure
subplot(1,4,1)
semilogy(G_vec,runtimes_G(2,:),'k--','LineWidth',0.01)
hold on
errorbar(G_vec,runtimes_G(1,:),errs_G(1,:),'k','LineWidth',0.8)
errorbar(G_vec,runtimes_G(2,:),errs_G(2,:),'Color',[0.9290 0.6940 0.1250],...
    'LineWidth',0.8)
errorbar(G_vec,runtimes_G(3,:),errs_G(3,:),'Color',[0.4940 0.1840 0.5560],...
    'LineWidth',0.8)
errorbar(G_vec,runtimes_G(4,:),errs_G(4,:),'Color',[0.3010 0.7450 0.9330],...
    'LineWidth',0.8)
legend({'','Central','SM','ADMM','ADMM-SC'},'location','southoutside','Interpreter','latex')
title('Runtime to Peak mCCR vs. G [Fixed N]','Interpreter','latex')
xlabel('G','Interpreter','latex')
ylabel('Runtime to Peak mCCR [s]','Interpreter','latex')
grid on
axis([7 53 1e-1 1e3])
xticks(G_vec)
yticks([1e-1 1e0 1e1 1e2 1e3])

subplot(1,4,2)
semilogy(G_vec,runtimes_G_N(2,:),'k--','LineWidth',0.01)
hold on
errorbar(G_vec,runtimes_G_N(1,:),errs_G_N(1,:),'k','LineWidth',0.8)
errorbar(G_vec,runtimes_G_N(2,:),errs_G_N(2,:),'Color',[0.9290 0.6940 0.1250],...
    'LineWidth',0.8)
errorbar(G_vec,runtimes_G_N(3,:),errs_G_N(3,:),'Color',[0.4940 0.1840 0.5560],...
    'LineWidth',0.8)
errorbar(G_vec,runtimes_G_N(4,:),errs_G_N(4,:),'Color',[0.3010 0.7450 0.9330],...
    'LineWidth',0.8)
title('Runtime to Peak mCCR vs. G [Increasing N]','Interpreter','latex')
xlabel('G','Interpreter','latex')
ylabel('Runtime to Peak mCCR [s]','Interpreter','latex')
grid on
axis([7 53 1e-1 1e3])
xticks(G_vec)
yticks([1e-1 1e0 1e1 1e2 1e3])

N_vec = [1000,1500,2000,2500,3000];
[accs_N,runtimes_N,adjs_N,errs_N,acc_errs_N] = comp_metrics(N_vec,N_exp,'N');
subplot(1,4,3)
semilogy(N_vec,runtimes_N(2,:),'k--','LineWidth',0.01)
hold on
errorbar(N_vec,runtimes_N(1,:),errs_N(1,:),'k','LineWidth',0.8)
errorbar(N_vec,runtimes_N(2,:),errs_N(2,:),'Color',[0.9290 0.6940 0.1250],...
    'LineWidth',0.8)
errorbar(N_vec,runtimes_N(3,:),errs_N(3,:),'Color',[0.4940 0.1840 0.5560],...
    'LineWidth',0.8)
errorbar(N_vec,runtimes_N(4,:),errs_N(4,:),'Color',[0.3010 0.7450 0.9330],...
    'LineWidth',0.8)
title('Runtime to Peak mCCR vs. N','Interpreter','latex')
xlabel('N','Interpreter','latex')
ylabel('Runtime to Peak mCCR [s]','Interpreter','latex')
grid on
axis([800 3200 1e-1 1e3])
xticks(N_vec)
yticks([1e-1 1e0 1e1 1e2 1e3])

P_vec = [4,6,8,10,12];
[accs_P,runtimes_P,adjs_P,errs_P,acc_errs_P] = comp_metrics(P_vec,N_exp,'P');
subplot(1,4,4)
semilogy(P_vec,runtimes_P(2,:),'k--','LineWidth',0.01)
hold on
errorbar(P_vec,runtimes_P(1,:),errs_P(1,:),'k','LineWidth',0.8)
errorbar(P_vec,runtimes_P(2,:),errs_P(2,:),'Color',[0.9290 0.6940 0.1250],...
    'LineWidth',0.8)
errorbar(P_vec,runtimes_P(3,:),errs_P(3,:),'Color',[0.4940 0.1840 0.5560],...
    'LineWidth',0.8)
errorbar(P_vec,runtimes_P(4,:),errs_P(4,:),'Color',[0.3010 0.7450 0.9330],...
    'LineWidth',0.8)
title('Runtime to Peak mCCR vs. P','Interpreter','latex')
xlabel('P','Interpreter','latex')
ylabel('Runtime to Peak mCCR [s]','Interpreter','latex')
grid on
axis([3 13 1e-1 1e3])
xticks(P_vec)
yticks([1e-1 1e0 1e1 1e2 1e3])

function [accs,runtimes,adjs,errs,acc_errs] = comp_metrics(vec,n_exps,str)
acc_vec_subgrad = zeros(1,length(vec));
acc_vec_ADMM = zeros(1,length(vec));
acc_vec_central = zeros(1,length(vec));
acc_vec_ADMM_sc = zeros(1,length(vec));

runtime_vec_subgrad = zeros(1,length(vec));
runtime_vec_ADMM = zeros(1,length(vec));
runtime_vec_central = zeros(1,length(vec));
runtime_vec_ADMM_sc = zeros(1,length(vec));

err_vec_subgrad = zeros(1,length(vec));
err_vec_ADMM = zeros(1,length(vec));
err_vec_central = zeros(1,length(vec));
err_vec_ADMM_sc = zeros(1,length(vec));

acc_err_vec_subgrad = zeros(1,length(vec));
acc_err_vec_ADMM = zeros(1,length(vec));
acc_err_vec_central = zeros(1,length(vec));
acc_err_vec_ADMM_sc = zeros(1,length(vec));

adj_vec_subgrad = zeros(1,length(vec));
adj_vec_ADMM = zeros(1,length(vec));
adj_vec_ADMM_sc = zeros(1,length(vec));

for i = 1:length(vec)
    acc_ten_subgrad_p = zeros([3,3,n_exps]);
    acc_ten_ADMM_p = zeros([3,3,n_exps]);
    acc_ten_central_p = zeros([5,5,n_exps]);
    acc_ten_ADMM_sc_p = zeros([3,3,n_exps]);

    runtime_ten_subgrad_p = zeros([3,3,n_exps]);
    runtime_ten_ADMM_p = zeros([3,3,n_exps]);
    runtime_ten_central_p = zeros([5,5,n_exps]);
    runtime_ten_ADMM_sc_p = zeros([3,3,n_exps]);

    adj_ten_subgrad_p = zeros([3,3,n_exps]);
    adj_ten_ADMM_p = zeros([3,3,n_exps]);
    adj_ten_ADMM_sc_p = zeros([3,3,n_exps]);
    for j = 1:n_exps
        filename = sprintf('FDR-SVM_Scalability_%s%d_%d.mat',str,vec(i),j);
        load(filename)
        acc_ten_subgrad_p(:,:,j) = acc_tensor_subgrad;
        runtime_ten_subgrad_p(:,:,j) = runtime_tensor_subgrad;
        adj_ten_subgrad_p(:,:,j) = runtime_tensor_subgrad - adj_tensor_subgrad;

        acc_ten_central_p(:,:,j) = acc_tensor_central;
        runtime_ten_central_p(:,:,j) = runtime_tensor_central;

        acc_ten_ADMM_p(:,:,j) = acc_tensor_ADMM;
        acc_ten_ADMM_sc_p(:,:,j) = acc_tensor_ADMM_sc;
    
        
        runtime_ten_ADMM_p(:,:,j) = runtime_tensor_ADMM;
        runtime_ten_ADMM_sc_p(:,:,j) = runtime_tensor_ADMM_sc;
        
        adj_ten_ADMM_p(:,:,j) = runtime_tensor_ADMM - adj_tensor_ADMM;
        adj_ten_ADMM_sc_p(:,:,j) = runtime_tensor_ADMM_sc - adj_tensor_ADMM_sc;
    end


    [acc_vec_subgrad(i),runtime_vec_subgrad(i),adj_vec_subgrad(i),err_vec_subgrad(i),acc_err_vec_subgrad(i)] = ...
        comp_opt_val(acc_ten_subgrad_p,runtime_ten_subgrad_p,adj_ten_subgrad_p);

    [acc_vec_ADMM(i),runtime_vec_ADMM(i),adj_vec_ADMM(i),err_vec_ADMM(i),acc_err_vec_ADMM(i)] = ...
        comp_opt_val(acc_ten_ADMM_p,runtime_ten_ADMM_p,adj_ten_ADMM_p);

    [acc_vec_ADMM_sc(i),runtime_vec_ADMM_sc(i),adj_vec_ADMM_sc(i),err_vec_ADMM_sc(i),acc_err_vec_ADMM_sc(i)] = ...
        comp_opt_val(acc_ten_ADMM_sc_p,runtime_ten_ADMM_sc_p,adj_ten_ADMM_sc_p);

    [acc_vec_central(i),runtime_vec_central(i),~,err_vec_central(i),acc_err_vec_central(i)] = ...
        comp_opt_val(acc_ten_central_p,runtime_ten_central_p,zeros([5,5]));
end
accs = [acc_vec_central;acc_vec_subgrad;acc_vec_ADMM;acc_vec_ADMM_sc];
runtimes = [runtime_vec_central;runtime_vec_subgrad;runtime_vec_ADMM;...
    runtime_vec_ADMM_sc];
adjs = [adj_vec_subgrad;adj_vec_ADMM;adj_vec_ADMM_sc];
errs = [err_vec_central;err_vec_subgrad;err_vec_ADMM;err_vec_ADMM_sc];
acc_errs = [acc_err_vec_central;acc_err_vec_subgrad;acc_err_vec_ADMM;acc_err_vec_ADMM_sc];
end

function [acc,runtime,adj,err,acc_err] = comp_opt_val(acc_ten,runtime_ten,adj_ten)
    acc_arr = round(mean(acc_ten,3),3);
    runtime_arr = mean(runtime_ten,3);
    adj_arr = mean(adj_ten,3);
    acc_max = max(max(acc_arr));
    [c,r] = find(acc_arr' == acc_max,1,'first');
    acc = acc_arr(r,c);
    runtime = runtime_arr(r,c);
    adj = adj_arr(r,c);
    err = std(runtime_ten(r,c,:));
    acc_err = std(acc_ten(r,c,:));
end