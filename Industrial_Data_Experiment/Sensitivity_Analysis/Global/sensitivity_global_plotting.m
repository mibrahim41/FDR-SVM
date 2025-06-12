N_exp = 50;
max_iter_vec = [5,10,20,60,100,140,180,220];
param_vec = logspace(-3,3,7);
[accs_iter,accs_param,acc_cent] = gen_final_result(N_exp,max_iter_vec,param_vec);
str_ids = ["nom","gro","cla","two","wro"];
titles = ["Nominal","Client Imabalnce","Class Imbalance","Client + Class Imbalance","Noisy Labels"];

[accs_iter_bds,accs_param_bds,acc_central_bds] = gen_bounds(N_exp,max_iter_vec,param_vec);

for i = 1:length(str_ids)
    err_temp_subgrad = std(accs_iter_bds(:,:,1,i),[],1);
    err_temp_ADMM = std(accs_iter_bds(:,:,2,i),[],1);
    err_temp_ADMM_sc = std(accs_iter_bds(:,:,3,i),[],1);
    err_temp_central = ones([1,length(max_iter_vec)])*std(acc_central_bds(i,:));

    acc_subgrad = accs_iter(1,:,i);
    acc_ADMM = accs_iter(2,:,i);
    acc_ADMM_sc = accs_iter(3,:,i);
    acc_central = ones([1,length(max_iter_vec)])*acc_cent(i);

    err_subgrad = zeros([2,length(err_temp_subgrad)]);
    err_ADMM = zeros([2,length(err_temp_ADMM)]);
    err_ADMM_sc = zeros([2,length(err_temp_ADMM_sc)]);
    err_central = zeros([2,length(err_temp_central)]);
    for j = 1:length(max_iter_vec)
        if err_temp_subgrad(j) + acc_subgrad(j) > 1
            err_subgrad(1,j) = 1 - acc_subgrad(j);
        else
            err_subgrad(1,j) = err_temp_subgrad(j);
        end
        err_subgrad(2,j) = err_temp_subgrad(j);

        if err_temp_ADMM(j) + acc_ADMM(j) > 1
            err_ADMM(1,j) = 1 - acc_ADMM(j);
        else
            err_ADMM(1,j) = err_temp_ADMM(j);
        end
        err_ADMM(2,j) = err_temp_ADMM(j);

        if err_temp_ADMM_sc(j) + acc_ADMM_sc(j) > 1
            err_ADMM_sc(1,j) = 1 - acc_ADMM_sc(j);
        else
            err_ADMM_sc(1,j) = err_temp_ADMM_sc(j);
        end
        err_ADMM_sc(2,j) = err_temp_ADMM_sc(j);

        if err_temp_central(j) + acc_central(j) > 1
            err_central(1,j) = 1 - acc_central(j);
        else
            err_central(1,j) = err_temp_central(j);
        end
        err_central(2,j) = err_temp_central(j);
    end


    subplot(2,length(str_ids),i)
    errorbar(max_iter_vec,accs_iter(1,:,i),err_subgrad(2,:),err_subgrad(1,:),...
        'Color',[0.9290 0.6940 0.1250],'LineWidth',0.8);
    hold on
    errorbar(max_iter_vec,ones([1,length(max_iter_vec)])*acc_cent(i),...
        err_central(2,:),err_central(1,:),'k--','LineWidth',0.8)
    errorbar(max_iter_vec,accs_iter(1,:,i),err_subgrad(2,:),err_subgrad(1,:),...
        'Color',[0.9290 0.6940 0.1250],'LineWidth',0.8)
    errorbar(max_iter_vec,accs_iter(2,:,i),err_ADMM(2,:),err_ADMM(1,:),...
        'Color',[0.4940 0.1840 0.5560],'LineWidth',0.8)
    errorbar(max_iter_vec,accs_iter(3,:,i),err_ADMM_sc(2,:),err_ADMM_sc(1,:),...
        'Color',[0.3010 0.7450 0.9330],'LineWidth',0.8)
    hold off
    grid on
    if i == 1
        legend({'','Central','SM','ADMM','ADMM-SC'},'Location','southoutside','Interpreter','latex')
    end
    xlabel('\# Communication Rounds T','Interpreter','latex')
    ylabel('mCCR','Interpreter','latex')
    title(titles(i),'Interpreter','latex')
    axis([0,240,0.5,1])

    err_temp_subgrad = std(accs_param_bds(:,:,1,i),[],1);
    err_temp_ADMM = std(accs_param_bds(:,:,2,i),[],1);
    err_temp_ADMM_sc = std(accs_param_bds(:,:,3,i),[],1);
    err_temp_central = ones([1,length(param_vec)])*std(acc_central_bds(i,:));

    acc_subgrad = accs_param(1,:,i);
    acc_ADMM = accs_param(2,:,i);
    acc_ADMM_sc = accs_param(3,:,i);
    acc_central = ones([1,length(param_vec)])*acc_cent(i);

    err_subgrad = zeros([2,length(err_temp_subgrad)]);
    err_ADMM = zeros([2,length(err_temp_ADMM)]);
    err_ADMM_sc = zeros([2,length(err_temp_ADMM_sc)]);
    err_central = zeros([2,length(err_temp_central)]);
    for j = 1:length(param_vec)
        if err_temp_subgrad(j) + acc_subgrad(j) > 1
            err_subgrad(1,j) = 1 - acc_subgrad(j);
        else
            err_subgrad(1,j) = err_temp_subgrad(j);
        end
        err_subgrad(2,j) = err_temp_subgrad(j);

        if err_temp_ADMM(j) + acc_ADMM(j) > 1
            err_ADMM(1,j) = 1 - acc_ADMM(j);
        else
            err_ADMM(1,j) = err_temp_ADMM(j);
        end
        err_ADMM(2,j) = err_temp_ADMM(j);

        if err_temp_ADMM_sc(j) + acc_ADMM_sc(j) > 1
            err_ADMM_sc(1,j) = 1 - acc_ADMM_sc(j);
        else
            err_ADMM_sc(1,j) = err_temp_ADMM_sc(j);
        end
        err_ADMM_sc(2,j) = err_temp_ADMM_sc(j);

        if err_temp_central(j) + acc_central(j) > 1
            err_central(1,j) = 1 - acc_central(j);
        else
            err_central(1,j) = err_temp_central(j);
        end
        err_central(2,j) = err_temp_central(j);
    end

    subplot(2,length(str_ids),i+length(str_ids))
    semilogx(param_vec,accs_param(1,:,i),'k--','LineWidth',0.01);
    hold on
    errorbar(param_vec,ones([1,length(param_vec)])*acc_cent(i),...
        err_central(2,:),err_central(1,:),'k--','LineWidth',0.8)
    errorbar(param_vec,accs_param(1,:,i),err_subgrad(2,:),err_subgrad(1,:),...
        'Color',[0.9290 0.6940 0.1250],'LineWidth',0.8);
    errorbar(param_vec,accs_param(2,:,i),err_ADMM(2,:),err_ADMM(1,:),...
        'Color',[0.4940 0.1840 0.5560],'LineWidth',0.8);
    errorbar(param_vec,accs_param(3,:,i),err_ADMM_sc(2,:),err_ADMM_sc(1,:),...
        'Color',[0.3010 0.7450 0.9330],'LineWidth',0.8);
    grid on
    xlabel('$\rho$,$\gamma$ Hyperparameter','Interpreter','latex')
    ylabel('mCCR','Interpreter','latex')
    title(titles(i),'Interpreter','latex')
    xticks(param_vec)
    axis([0,1600,0.5,1])
end

function [accs_iter,accs_param,acc_cent] = gen_final_result(N_exp,m_vec,p_vec)
str_ids = ["nom","gro","cla","two","wro"];
accs_iter = zeros(3,length(m_vec),length(str_ids));
accs_param = zeros(3,length(p_vec),length(str_ids));
acc_cent = zeros([1,length(str_ids)]);
for i = 1:length(str_ids)
    [acc_mat_subgrad,acc_mat_ADMM,acc_mat_ADMM_sc,acc_mat_central] = ind_exp(N_exp,m_vec,p_vec,str_ids(i));
    [accs_iter(1,:,i),accs_param(1,:,i)] = extract_vecs(acc_mat_subgrad);
    [accs_iter(2,:,i),accs_param(2,:,i)] = extract_vecs(acc_mat_ADMM);
    [accs_iter(3,:,i),accs_param(3,:,i)] = extract_vecs(acc_mat_ADMM_sc);
    acc_cent(i) = max(max(acc_mat_central));
end
end

function [acc_mat_subgrad,acc_mat_ADMM,acc_mat_ADMM_sc,acc_mat_central] = ind_exp(N_exp,m_vec,p_vec,str_id)
acc_mat_subgrad = zeros([length(m_vec),length(p_vec)]);
acc_mat_ADMM = zeros([length(m_vec),length(p_vec)]);
acc_mat_ADMM_sc = zeros([length(m_vec),length(p_vec)]);
acc_mat_central = zeros([5,5]);
for i = 1:N_exp
    filename = sprintf('FDR-SVM_sensitivity_%s_%d.mat',str_id,i);
    load(filename)
    acc_mat_subgrad = acc_mat_subgrad + acc_tensor_subgrad;
    acc_mat_ADMM = acc_mat_ADMM + acc_tensor_ADMM;
    acc_mat_ADMM_sc = acc_mat_ADMM_sc + acc_tensor_ADMM_sc;
    acc_mat_central = acc_mat_central + acc_tensor_central;
end
acc_mat_subgrad = (1/N_exp)*acc_mat_subgrad;
acc_mat_ADMM = (1/N_exp)*acc_mat_ADMM;
acc_mat_ADMM_sc = (1/N_exp)*acc_mat_ADMM_sc;
acc_mat_central = (1/N_exp)*acc_mat_central;
end

function [acc_vec_iter,acc_vec_param] = extract_vecs(acc_mat)
max_acc = max(max(acc_mat));
[c,r] = find(acc_mat' == max_acc,1,'first');
acc_vec_iter = acc_mat(:,c);
acc_vec_param = acc_mat(r,:);
end

function [accs_iter_bds,accs_param_bds,acc_central_bds] = gen_bounds(N_exp,m_vec,p_vec)
str_ids = ["nom","gro","cla","two","wro"];
accs_iter_bds = zeros(N_exp,length(m_vec),4,length(str_ids));
accs_param_bds = zeros(N_exp,length(p_vec),4,length(str_ids));
acc_central_bds = zeros(length(str_ids),N_exp);
for i = 1:length(str_ids)
    [acc_ten_subgrad,acc_ten_ADMM,acc_ten_ADMM_sc,acc_ten_central] = all_exp(N_exp,m_vec,p_vec,str_ids(i));
    [accs_iter_bds(:,:,1,i),accs_param_bds(:,:,1,i)] = extract_bds(acc_ten_subgrad);
    [accs_iter_bds(:,:,2,i),accs_param_bds(:,:,2,i)] = extract_bds(acc_ten_ADMM);
    [accs_iter_bds(:,:,3,i),accs_param_bds(:,:,3,i)] = extract_bds(acc_ten_ADMM_sc);

    acc_mat_central = mean(acc_ten_central,3);
    [r,c,~] = find(acc_mat_central == max(max(acc_mat_central)),1,'first');
    acc_central_bds(i,:) = acc_ten_central(r,c,:);
end
end

function [acc_ten_subgrad,acc_ten_ADMM,acc_ten_ADMM_sc,acc_ten_central] = all_exp(N_exp,m_vec,p_vec,str_id)
acc_ten_subgrad = zeros([length(m_vec),length(p_vec),N_exp]);
acc_ten_ADMM = zeros([length(m_vec),length(p_vec),N_exp]);
acc_ten_ADMM_sc = zeros([length(m_vec),length(p_vec),N_exp]);
acc_ten_central = zeros([5,5,N_exp]);
for i = 1:N_exp
    filename = sprintf('FDR-SVM_sensitivity_%s_%d.mat',str_id,i);
    load(filename)
    acc_ten_subgrad(:,:,i) = acc_tensor_subgrad;
    acc_ten_ADMM(:,:,i) = acc_tensor_ADMM;
    acc_ten_ADMM_sc(:,:,i) = acc_tensor_ADMM_sc;
    acc_ten_central(:,:,i) = acc_tensor_central;
end
end

function [acc_vec_iter_bds,acc_vec_param_bds] = extract_bds(acc_ten)
[row,col,l] = size(acc_ten);
acc_mat = mean(acc_ten,3);
max_acc = max(max(acc_mat));
[c,r] = find(acc_mat' == max_acc,1,'first');
acc_vec_iter_bds = zeros([l,row]);
acc_vec_param_bds = zeros([l,col]);
for i = 1:row
    acc_vec_iter_bds(:,i) = acc_ten(i,c,:);
end

for j = 1:col
    acc_vec_param_bds(:,j) = acc_ten(r,j,:);
end
end