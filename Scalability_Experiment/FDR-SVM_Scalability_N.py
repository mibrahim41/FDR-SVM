import gurobipy as grb
import numpy as np
import timeit
import scipy
from sklearn.datasets import make_classification
import sklearn

# Setting Up Data Generation    
def generate_n_sample_vec(weights,total_samples):
    n_classes = len(weights)
    n_samples_vec = np.zeros(len(weights))
    for j in range(len(n_samples_vec)):
        n_samples_vec[j] = np.round(weights[j]*total_samples)
        
    n_samples_vec = n_samples_vec.astype(int)
    
    count_add = 0
    count_subt = 0
    while sum(n_samples_vec) < total_samples:
        idx_add = count_add%n_classes
        n_samples_vec[idx_add] += 1
        count_add += 1
        
    while sum(n_samples_vec) > total_samples:
        idx_subt = count_subt%n_classes
        n_samples_vec[idx_subt] -= 1
        count_subt += 1
        
    return n_samples_vec

def normalize_data(data_in,lim):
    row,col = data_in.shape
    data_norm = np.zeros([row,col])
    for r in range(row):
        tmp = data_in[r]
        data_norm[r] = (tmp - -lim)/(lim - -lim)
    
    return data_norm

def change_labels(perc_wrong_y,total_samples,y):
    n_wrong = round(perc_wrong_y*total_samples)
    idx_wrong = np.random.randint(0, high=total_samples, size=n_wrong)
    y[idx_wrong] = y[idx_wrong]*-1
    
    return y
    
def gen_data(G,n_train, n_test, n_features, n_informative, n_redundant, 
             weights_train, weights_test, perc_wrong_y=0, class_sep=1.5, n_clusters=1,perc_g_vec=-1,lim=10):
    
    # Obtain an upper bound on total number of samples
    tot_samples = (np.max(weights_train)*2*n_train + np.max(weights_test)*2*n_test)*10
    
    # Generate simulation data
    x_all, y_all = make_classification(n_samples=int(tot_samples),
                                       n_features=n_features,
                                       n_informative=n_informative, 
                                       n_redundant=n_redundant, 
                                       n_repeated=0, 
                                       n_classes=2, 
                                       n_clusters_per_class=n_clusters, 
                                       weights=None, 
                                       flip_y=0, 
                                       class_sep=class_sep, 
                                       hypercube=True, 
                                       shift=0.0, 
                                       scale=1.0, 
                                       shuffle=False, 
                                       random_state=None)
    
    
    i = 0
    while i < len(x_all):
        if np.any(x_all[i] > lim) or np.any(x_all[i] < -lim):
            x_all = np.delete(x_all, (i), axis=0)
            y_all = np.delete(y_all, (i), axis=0)
        else:
            i += 1
        
        
    # Separate test set
    x_test = np.zeros([n_test,n_features])
    y_test = np.zeros(n_test)
    
    test_idx_vec = generate_n_sample_vec(weights_test,n_test)
    
    x_rem = np.zeros([len(x_all) - sum(test_idx_vec),n_features])
    y_rem = np.zeros(len(y_all) - sum(test_idx_vec))
    
    prev_idx = 0
    for c in range(2):
        x_c = x_all[y_all == c,:]
        n_samples_c = test_idx_vec[c]
        x_test[n_samples_c*c:n_samples_c*c + n_samples_c,:] = x_c[:n_samples_c,:]
        y_test[n_samples_c*c:n_samples_c*c + n_samples_c] = c
        
        idx_rem = len(x_c) - test_idx_vec[c]
        x_rem[prev_idx:len(x_c) - test_idx_vec[c] + prev_idx,:] = x_c[test_idx_vec[c]:,:]
        y_rem[prev_idx:len(x_c) - test_idx_vec[c] + prev_idx] = c
        
        prev_idx = len(x_c) - test_idx_vec[c] + prev_idx

    y_test = y_test.astype(int)
    y_test[y_test == 0] = -1
    
    x_test_norm = normalize_data(x_test,lim)
    
    # Creating Client Training Sets
    if perc_g_vec == -1:
        tmp_perc_vec = np.random.uniform(low=0.1,high=1.0,size=G)
        perc_g_vec = tmp_perc_vec/np.sum(tmp_perc_vec)
        
    elif perc_g_vec == 0:
        perc_g_vec = np.ones(G)*(1/G)
        
        
    Ng_vec = generate_n_sample_vec(perc_g_vec,n_train)
    
    x_central = np.zeros([np.sum(Ng_vec),n_features])
    x_central_norm = np.zeros([np.sum(Ng_vec),n_features])
    y_central = np.zeros(np.sum(Ng_vec))
    
    client_sets = {}
    client_sets_norm = {}
    
    x_all = x_rem
    y_all = y_rem
    
    for g in range(len(Ng_vec)):
        x_train_g = np.zeros([Ng_vec[g],n_features])
        y_train_g = np.zeros(Ng_vec[g])

        train_g_idx_vec = generate_n_sample_vec(weights_train,Ng_vec[g])

        x_rem = np.zeros([len(x_all) - sum(train_g_idx_vec),n_features])
        y_rem = np.zeros(len(y_all) - sum(train_g_idx_vec))
        
        prev_idx = 0
        for c in range(2):
            x_c = x_all[y_all == c,:]
            n_samples_c_g = train_g_idx_vec[c]
            x_train_g[n_samples_c_g*c:n_samples_c_g*c + n_samples_c_g,:] = x_c[:n_samples_c_g,:]
            y_train_g[n_samples_c_g*c:n_samples_c_g*c + n_samples_c_g] = c
            
            idx_rem = len(x_c) - train_g_idx_vec[c]
            x_rem[prev_idx:len(x_c) - train_g_idx_vec[c] + prev_idx,:] = x_c[train_g_idx_vec[c]:,:]
            y_rem[prev_idx:len(x_c) - train_g_idx_vec[c] + prev_idx] = c
            
            prev_idx = len(x_c) - train_g_idx_vec[c] + prev_idx
            
        y_train_g[y_train_g == 0] = -1
        x_train_g_norm = normalize_data(x_train_g,lim)
        
        if perc_wrong_y > 0:
            y_train_g = change_labels(perc_wrong_y,len(y_train_g),y_train_g)

        client_sets['x'+str(g)] = x_train_g
        client_sets['y'+str(g)] = y_train_g

        client_sets_norm['x'+str(g)] = x_train_g_norm
        client_sets_norm['y'+str(g)] = y_train_g
        
        if g == 0:
            x_central[0:Ng_vec[g],:] = x_train_g
            x_central_norm[0:Ng_vec[g],:] = x_train_g_norm
        
            y_central[0:Ng_vec[g]] = y_train_g
        elif g > 0:
            x_central[np.sum(Ng_vec[:g]):Ng_vec[g]+np.sum(Ng_vec[:g]),:] = x_train_g
            x_central_norm[np.sum(Ng_vec[:g]):Ng_vec[g]+np.sum(Ng_vec[:g]),:] = x_train_g_norm
        
            y_central[np.sum(Ng_vec[:g]):Ng_vec[g]+np.sum(Ng_vec[:g])] = y_train_g
        
        x_all = x_rem
        y_all = y_rem
        
    central_set = {'x': x_central, 'y': y_central}
    central_set_norm = {'x': x_central_norm, 'y': y_central}
    
    test_set = {'x': x_test, 'y': y_test}
    test_set_norm = {'x': x_test_norm, 'y': y_test}
            
        
    
    return central_set, client_sets, test_set, central_set_norm, client_sets_norm, test_set_norm, Ng_vec
        

class ADMM_alg:
    """ADMM-Based Optimization Algorithm"""
    
    def __init__(self,param,init_vars,Ng_vec):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        self.rho = param['rho']
        self.max_iter = param['max_iter']
        self.tau = param['tau']
        self.w_global = init_vars['w']
        self.mu = init_vars['mu']
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
        self.Ng_vec = Ng_vec
        
    def train_clients(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        self.w_local = np.zeros([self.G,col_sample])
        runtime_vec = np.zeros(self.G)
        
        for g in range(self.G):
            train_data_g = {'x':train_data['x'+str(g)], 'y':train_data['y'+str(g)]}
            start_g = timeit.default_timer()
            optimal_g = self.client_DR_SVM(train_data_g,g)
            stop_g = timeit.default_timer()
            self.w_local[g,:] = optimal_g['w']
            runtime_vec[g] = stop_g - start_g
            
        return runtime_vec

        
    def client_DR_SVM(self,data,g):
        """Distributionally robust binary SVM without support information"""
        
        x_train = data['x']
        y_train = data['y'].flatten()

        row, col = x_train.shape
        optimal = {}
        

        # Step 0: create model
        model = grb.Model('DRSVM_without_support_'+str(g))
        model.setParam('OutputFlag', False)

        # Step 1: define decision variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_w = {}
        slack_var = {}
        for n in range(row):
            var_s[n] = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0)
        for p in range(col):
            var_w[p] = model.addVar(
                vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
            if self.pnorm == 1:
                slack_var[p] = model.addVar(vtype=grb.GRB.CONTINUOUS)

        # Step 2: integerate variables
        model.update()

        # Step 3: define constraints
        for n in range(row):
            model.addConstr(
                1 - y_train[n] * grb.quicksum(var_w[p] * x_train[n,p]
                                              for p in range(col)) <= var_s[n])
            model.addConstr(
                1 + y_train[n] * grb.quicksum(var_w[p] * x_train[n, p]
                                              for p in range(col)) -
                self.kappa[g] * var_lambda <= var_s[n])

        if self.pnorm == 1:
            for p in range(col):
                model.addConstr(var_w[p] <= slack_var[p])
                model.addConstr(-var_w[p] <= slack_var[p])
            model.addConstr(grb.quicksum(slack_var[p]
                                         for p in range(col)) <= var_lambda)
        elif self.pnorm == 2:
            model.addQConstr(
                grb.quicksum(var_w[p] * var_w[p]
                             for p in range(col)) <= var_lambda * var_lambda)

        elif self.pnorm == float('Inf'):
            for p in range(col):
                model.addConstr(var_w[p] <= var_lambda)
                model.addConstr(-var_w[p] <= var_lambda)

        # Step 4: define objective value
        sum_var_s = grb.quicksum(var_s[n] for n in range(row))
        norm_regularizer = grb.quicksum((var_w[p] - self.w_global[p] + self.mu[g][p])*\
                                       (var_w[p] - self.w_global[p] + self.mu[g][p]) for p in range(col))
        obj = var_lambda*self.epsilon[g] + (1/row)*sum_var_s + (self.rho/2)*norm_regularizer + self.tau*grb.quicksum(var_w[p]*var_w[p] for p in range(col))
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[p].x for p in range(col)])
        tmp = {'w': w_opt,'objective': model.ObjVal,'diagnosis': model.status}
        optimal.update(tmp)
        
        model.close()
        return optimal
    
    def master_comp(self):
        temp_w_global = np.zeros(np.shape(self.w_global))
        for g in range(self.G):
            temp_w_global = temp_w_global + self.alphag_vec[g]*(self.w_local[g]+self.mu[g])
        
        self.w_global = temp_w_global
        
    def update_mu(self):
        runtime_vec = np.zeros(self.G)
        for g in range(self.G):
            start_g = timeit.default_timer()
            self.mu[g,:] = self.mu[g,:] + self.w_local[g,:] - self.w_global
            stop_g = timeit.default_timer()
            runtime_vec[g] = stop_g - start_g
            
        return runtime_vec
            
            
    def train(self,train_data):
        runtime_vec = np.zeros(self.G)
        for t in range(self.max_iter):
            runtime_vec_A_t = self.train_clients(train_data)
            runtime_vec = runtime_vec + runtime_vec_A_t
            self.master_comp()
            runtime_vec_B_t = self.update_mu()
            runtime_vec = runtime_vec + runtime_vec_B_t
            
        adj = np.sum(runtime_vec) - np.max(runtime_vec)
        return self.w_global, adj

    
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape
        
        y_pred = np.ones(np.shape(y_test))
        
        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
        acc = 1-np.sum(y_pred != y_test)/len(y_test)
        
        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
            

class subgradient_alg:
    """Subgradient-based algorithm for traininf distributionally robust SVMs"""
    
    def __init__(self,param,init_vars,Ng_vec):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        self.max_iter = param['max_iter']
        self.stepsize = param['stepsize']
        self.w_global = init_vars['w']
        self.Ng_vec = Ng_vec
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
            
    def extremal_dist(self,data,g):
        
        x_train = data['x']
        y_train = data['y']

        row, col = x_train.shape
        optimal = {}
        
        # Step 0: create model
        model = grb.Model('Extremal_Distribution_SVM_'+str(g))
        model.setParam('OutputFlag', False)
        
        # Step 1: define decision variables
        var_b_pos = {}
        var_b_neg = {}
        var_q_pos = {}
        var_q_neg = {}
        slack_q_pos = {}
        slack_q_neg = {}
        aux_q_pos = {}
        aux_q_neg = {}
        slack_b_pos = {}
        slack_b_neg = {}
        
        for n in range(row):
            var_b_pos[n] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=1e-5)
            var_b_neg[n] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=1e-5)
            
        for n in range(row):
            for p in range(col):
                var_q_pos[n,p] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
                var_q_neg[n,p] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
                if self.pnorm == 1:
                    slack_q_pos[n,p] = model.addVar(vtype=grb.GRB.CONTINUOUS)
                    slack_q_neg[n,p] = model.addVar(vtype=grb.GRB.CONTINUOUS)
            if self.pnorm == float('Inf'):
                aux_q_pos[n] = model.addVar(vtype=grb.GRB.CONTINUOUS)
                aux_q_neg[n] = model.addVar(vtype=grb.GRB.CONTINUOUS)
                
                    
                
        # Step 2: integrate variables
        model.update()
        
        # Step3: define constraints
        if self.pnorm == 1:
            for p in range(col):
                for n in range(row):
                    model.addConstr(var_q_pos[n,p] <= slack_q_pos[n,p])
                    model.addConstr(-var_q_pos[n,p] <= slack_q_pos[n,p])
                    model.addConstr(var_q_neg[n,p] <= slack_q_neg[n,p])
                    model.addConstr(-var_q_neg[n,p] <= slack_q_neg[n,p])
            model.addConstr(grb.quicksum(\
                grb.quicksum(slack_q_pos[n,p] for p in range(col)) + grb.quicksum(slack_q_neg[n,p] for p in range(col)) +\
                self.kappa[g]*var_b_neg[n] for n in range(row)) <= row*self.epsilon[g])

        elif self.pnorm == float('Inf'):
            for p in range(col):
                for n in range(row):
                    model.addConstr(var_q_pos[n,p] <= aux_q_pos[n])
                    model.addConstr(-var_q_pos[n,p] <= aux_q_pos[n])
                    model.addConstr(var_q_neg[n,p] <= aux_q_neg[n])
                    model.addConstr(-var_q_neg[n,p] <= aux_q_neg[n])
                    
            model.addConstr(grb.quicksum(aux_q_pos[n] + aux_q_neg[n] + self.kappa[g]*var_b_neg[n] for n in range(row)) <=\
                           row*self.epsilon[g])
        
        for n in range(row):
            model.addConstr(var_b_pos[n] + var_b_neg[n] == 1)
            
            
        for n in range(row):
            for p in range(col):
                model.addConstr(var_b_pos[n]*(x_train[n,p] - 1) - var_q_pos[n,p] <= 0)
                model.addConstr(var_b_pos[n]*x_train[n,p] - var_q_pos[n,p] >= 0)
                model.addConstr(var_b_neg[n]*(x_train[n,p] - 1) - var_q_neg[n,p] <= 0)
                model.addConstr(var_b_neg[n]*x_train[n,p] - var_q_neg[n,p] >= 0)
            
        # Step 4: define objective value
        temp_sum = grb.quicksum(\
            (var_b_pos[n] - var_b_neg[n])*(-y_train[n])*grb.quicksum(self.w_global[p]*x_train[n,p] for p in range(col))\
            + (-y_train[n])*grb.quicksum(self.w_global[p]*(var_q_pos[n,p] - var_q_neg[n,p]) for p in range(col))\
                                for n in range(row))
        obj = (1/row)*temp_sum
        model.setObjective(obj, grb.GRB.MAXIMIZE)
        
        # Step 5: solve the problem
        model.optimize()
        
        # Step 6: store results
        b_pos = np.ones([row])
        b_neg = np.ones([row])
        q_pos = np.ones([row,col])
        q_neg = np.ones([row,col])
        
        for n in range(row):
            b_pos[n] = var_b_pos[n].x
            b_neg[n] = var_b_neg[n].x
            for p in range(col):
                q_pos[n,p] = var_q_pos[n,p].x
                q_neg[n,p] = var_q_neg[n,p].x
                
        temp = {'b_pos': b_pos, 'b_neg': b_neg, 'q_pos': q_pos, 'q_neg': q_neg, 'objective': model.ObjVal, 
                'diagnosis': model.status}
        optimal.update(temp)
        
        model.close()
        return optimal
        
    def subgrad_comp(self,data,opt_dict):
        x_train = data['x']
        y_train = data['y']

        b_pos = opt_dict['b_pos']
        b_neg = opt_dict['b_neg']
        q_pos = opt_dict['q_pos']
        q_neg = opt_dict['q_neg']

        row,col = q_pos.shape

        temp_A = np.zeros([row,col])
        temp_B = np.zeros([row,col])

        for n in range(row):
            z_pos = x_train[n,:] - (1/b_pos[n])*q_pos[n,:]
            z_neg = x_train[n,:] - (1/b_neg[n])*q_neg[n,:]

            if 1 - y_train[n]*np.matmul(self.w_global,z_pos) > 0:
                temp_A[n,:] = -b_pos[n]*y_train[n]*z_pos

            if 1 + y_train[n]*np.matmul(self.w_global,z_neg) > 0:
                temp_B[n,:] = b_neg[n]*y_train[n]*z_neg

        temp_sum_A = np.sum(temp_A,axis=0)
        temp_sum_B = np.sum(temp_B,axis=0)
        subgrad = (1/row)*(temp_sum_A + temp_sum_B)

        return subgrad

    def master_comp(self,all_subgrads):
        agg_subgrad = np.zeros([len(all_subgrads[0,:])])
        for g in range(self.G):
            agg_subgrad = agg_subgrad + self.alphag_vec[g]*all_subgrads[g,:]

        return agg_subgrad
            
    def train_clients(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        all_subgrads = np.zeros([self.G,col_sample])
        runtime_vec = np.zeros(self.G)
        
        for g in range(self.G):
            train_data_g = {'x':train_data['x'+str(g)], 'y':train_data['y'+str(g)]}
            start_g = timeit.default_timer()
            optimal_g = self.extremal_dist(train_data_g,g)
            subgrad_g = self.subgrad_comp(train_data_g,optimal_g)
            stop_g = timeit.default_timer()
            all_subgrads[g,:] = subgrad_g
            runtime_vec[g] = stop_g - start_g

        return all_subgrads, runtime_vec
        
    def train(self,train_data):
        runtime_vec = np.zeros(self.G)
        for t in range(self.max_iter):
            all_subgrads,runtime_vec_t = self.train_clients(train_data)
            runtime_vec = runtime_vec + runtime_vec_t
            agg_subgrad = self.master_comp(all_subgrads)
            self.w_global = self.w_global - (self.stepsize/np.sqrt(t+1))*(agg_subgrad)
            
        adj = np.sum(runtime_vec) - np.max(runtime_vec)
        return self.w_global, adj
        
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape

        y_pred = np.ones(np.shape(y_test))

        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1

        acc = 1-np.sum(y_pred != y_test)/len(y_test)

        return y_pred,acc

    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""

        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)

        return conf_mat,disp_conf_mat


class centralized_classifier:
    
    def __init__(self,param):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        
    def train(self,data):
        x_train = data['x']
        y_train = data['y'].flatten()

        row, col = x_train.shape
        optimal = {}


        # Step 0: create model
        model = grb.Model('DRSVM_without_support_')
        model.setParam('OutputFlag', False)

        # Step 1: define decision variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_w = {}
        slack_var = {}
        for n in range(row):
            var_s[n] = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0)
        for p in range(col):
            var_w[p] = model.addVar(
                vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
            if self.pnorm == 1:
                slack_var[p] = model.addVar(vtype=grb.GRB.CONTINUOUS)

        # Step 2: integerate variables
        model.update()

        # Step 3: define constraints
        for n in range(row):
            model.addConstr(
                1 - y_train[n] * grb.quicksum(var_w[p] * x_train[n,p]
                                              for p in range(col)) <= var_s[n])
            model.addConstr(
                1 + y_train[n] * grb.quicksum(var_w[p] * x_train[n, p]
                                              for p in range(col)) -
                self.kappa * var_lambda <= var_s[n])

        if self.pnorm == 1:
            for p in range(col):
                model.addConstr(var_w[p] <= slack_var[p])
                model.addConstr(-var_w[p] <= slack_var[p])
            model.addConstr(grb.quicksum(slack_var[p]
                                         for p in range(col)) <= var_lambda)
        elif self.pnorm == 2:
            model.addQConstr(
                grb.quicksum(var_w[p] * var_w[p]
                             for p in range(col)) <= var_lambda * var_lambda)

        elif self.pnorm == float('Inf'):
            for p in range(col):
                model.addConstr(var_w[p] <= var_lambda)
                model.addConstr(-var_w[p] <= var_lambda)

        # Step 4: define objective value
        sum_var_s = grb.quicksum(var_s[n] for n in range(row))
        obj = var_lambda*self.epsilon + (1/row)*sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[p].x for p in range(col)])
        tmp = {'w': w_opt,'objective': model.ObjVal,'diagnosis': model.status}
        optimal.update(tmp)
        self.w_opt = w_opt

        model.close()
        return self.w_opt

    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape
        
        y_pred = np.ones(np.shape(y_test))
        
        for n in range(row_x):
            pred_score = np.sum(self.w_opt*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
        acc = 1-np.sum(y_pred != y_test)/len(y_test)
        
        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat


def save_exp_file(P,G,N_train,N_test,weights_train,weights_test,class_sep,perc_g_vec,num_exp,max_iter_vec_ADMM,max_iter_vec_subgrad,
                 param_vec_ADMM,param_vec_subgrad,epsilon_vec,kappa_vec,acc_tensor_ADMM_sc,acc_tensor_ADMM,
                 runtime_tensor_ADMM_sc,runtime_tensor_ADMM,acc_tensor_central,
                 runtime_tensor_central,adj_tensor_ADMM_sc,adj_tensor_ADMM):
    exp_dict = {
        'P':P,
        'G':G,
        'N_train':N_train,
        'N_test':N_test,
        'weights_train':weights_train,
        'weights_test':weights_test,
        'class_sep':class_sep,
        'perc_g_vec':perc_g_vec,
        'num_exp':num_exp,
        'max_iter_vec_ADMM':max_iter_vec_ADMM,
        'param_vec_ADMM':param_vec_ADMM,
        'max_iter_vec_subgrad':max_iter_vec_subgrad,
        'param_vec_subgrad':param_vec_subgrad,
        'epsilon_vec':epsilon_vec,
        'kappa_vec':kappa_vec,
        'acc_tensor_ADMM_sc':acc_tensor_ADMM_sc,
        'acc_tensor_ADMM':acc_tensor_ADMM,
        'runtime_tensor_ADMM_sc':runtime_tensor_ADMM_sc,
        'runtime_tensor_ADMM':runtime_tensor_ADMM,
        'acc_tensor_central':acc_tensor_central,
        'runtime_tensor_central':runtime_tensor_central,
        'adj_tensor_ADMM_sc':adj_tensor_ADMM_sc,
        'adj_tensor_ADMM':adj_tensor_ADMM,
    }

    filename = 'FDR-SVM_Scalability_ADMM_N'+str(N_train)+'_'+str(int(timeit.default_timer()))+str(np.random.randint(0,high=1000))+'.mat'
    scipy.io.savemat(filename,exp_dict)


P = 4
G = 10
N_train_vec = [1000,1500,2000,2500,3000]
N_test = 2000
weights_train = [0.5,0.5]
weights_test = [0.5,0.5]
class_sep = 1.2
perc_g_vec = list(np.ones(G)*(1/G))
tau_x = 270


num_exp = 1
max_iter_vec_subgrad = [5,140,180,220]
max_iter_vec_ADMM = [10,20,30]
param_vec_subgrad = np.logspace(1,3,3)
param_vec_ADMM = np.logspace(-3,-1,3)

acc_tensor_subgrad = np.zeros([len(max_iter_vec_subgrad),len(param_vec_subgrad),num_exp])
acc_tensor_ADMM = np.zeros([len(max_iter_vec_ADMM),len(param_vec_ADMM)])
acc_tensor_ADMM_sc = np.zeros([len(max_iter_vec_ADMM),len(param_vec_ADMM)])

runtime_tensor_subgrad = np.zeros([len(max_iter_vec_subgrad),len(param_vec_subgrad),num_exp])
runtime_tensor_ADMM = np.zeros([len(max_iter_vec_ADMM),len(param_vec_ADMM)])
runtime_tensor_ADMM_sc = np.zeros([len(max_iter_vec_ADMM),len(param_vec_ADMM)])

adj_tensor_subgrad = np.zeros([len(max_iter_vec_subgrad),len(param_vec_subgrad),num_exp])
adj_tensor_ADMM = np.zeros([len(max_iter_vec_ADMM),len(param_vec_ADMM)])
adj_tensor_ADMM_sc = np.zeros([len(max_iter_vec_ADMM),len(param_vec_ADMM)])
        
        
epsilon_vec = np.logspace(-5,-1,5)
kappa_vec = np.asarray([0.1,0.25,0.5,0.75,1])

acc_tensor_central = np.zeros([len(epsilon_vec),len(kappa_vec),num_exp])
runtime_tensor_central = np.zeros([len(epsilon_vec),len(kappa_vec),num_exp])




for i in range(num_exp):
    for N_train in N_train_vec:
    
        central_set, client_sets, test_set, central_set_norm, client_sets_norm, test_set_norm, Ng_vec = gen_data(G=G,
        n_train=N_train, n_test=N_test, n_features=P, n_informative=P, n_redundant=0, weights_train=weights_train, 
        weights_test=weights_test, perc_wrong_y=0, class_sep=class_sep, n_clusters=1,
        perc_g_vec=perc_g_vec, lim=10)
        
        epsilon_vec_algs = np.zeros(G)
        for j in range(len(Ng_vec)):
            epsilon_vec_algs[j] = 1/(10*Ng_vec[j])
        
        
        for s in range(len(epsilon_vec)):
            for k in range(len(kappa_vec)):
                params_central = {'epsilon':epsilon_vec[s], 'kappa':kappa_vec[k], 'pnorm':1}
                
                classifier_central = centralized_classifier(params_central)
                start_central = timeit.default_timer()
                w_opt_central = classifier_central.train(central_set)
                stop_central = timeit.default_timer()
                y_pred_central, acc_central = classifier_central.test(test_set)

                acc_tensor_central[s,k,i] = acc_central
                runtime_tensor_central[s,k,i] = stop_central - start_central

        for m in range(len(max_iter_vec_ADMM)):
            for p in range(len(param_vec_ADMM)):
                params_subgrad = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*0.25, 'pnorm':float('Inf'), 
                                'max_iter':max_iter_vec_subgrad[m], 'stepsize':param_vec_subgrad[p]}
                init_vars_subgrad = {'w':np.zeros(P)}


                params_ADMM = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*0.25, 'pnorm':1,
                            'rho':param_vec_ADMM[p], 'max_iter':max_iter_vec_ADMM[m], 'tau':0}
                init_vars_ADMM = {'w':np.zeros(P), 'mu':np.ones([G,P])}

                params_ADMM_sc = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*0.25, 'pnorm':1,
                            'rho':param_vec_ADMM[p], 'max_iter':max_iter_vec_ADMM[m], 'tau':tau_x*p}

                classifier_subgrad = subgradient_alg(params_subgrad, init_vars_subgrad, Ng_vec)
                start_subgrad = timeit.default_timer()
                w_opt_subgrad,adj_subgrad = classifier_subgrad.train(client_sets_norm)
                stop_subgrad = timeit.default_timer()
                y_pred_subgrad, acc_subgrad = classifier_subgrad.test(test_set_norm)
                acc_tensor_subgrad[m,p,i] = acc_subgrad
                runtime_tensor_subgrad[m,p,i] = stop_subgrad - start_subgrad
                adj_tensor_subgrad[m,p,i] = adj_subgrad

                classifier_ADMM = ADMM_alg(params_ADMM, init_vars_ADMM, Ng_vec)
                start_ADMM = timeit.default_timer()
                w_opt_ADMM,adj_ADMM = classifier_ADMM.train(client_sets)
                stop_ADMM = timeit.default_timer()
                y_pred_ADMM, acc_ADMM = classifier_ADMM.test(test_set)
                acc_tensor_ADMM[m,p] = acc_ADMM
                runtime_tensor_ADMM[m,p] = stop_ADMM - start_ADMM
                adj_tensor_ADMM[m,p] = adj_ADMM

                classifier_ADMM_sc = ADMM_alg(params_ADMM_sc, init_vars_ADMM, Ng_vec)
                start_ADMM_sc = timeit.default_timer()
                w_opt_ADMM_sc,adj_ADMM_sc = classifier_ADMM_sc.train(client_sets)
                stop_ADMM_sc = timeit.default_timer()
                y_pred_ADMM_sc, acc_ADMM_sc = classifier_ADMM_sc.test(test_set)
                acc_tensor_ADMM_sc[m,p] = acc_ADMM_sc
                runtime_tensor_ADMM_sc[m,p] = stop_ADMM_sc - start_ADMM_sc
                adj_tensor_ADMM_sc[m,p] = adj_ADMM_sc

        save_exp_file(P,G,N_train,N_test,weights_train,weights_test,class_sep,perc_g_vec,num_exp,max_iter_vec_ADMM,max_iter_vec_subgrad,
                 param_vec_ADMM,param_vec_subgrad,epsilon_vec,kappa_vec,acc_tensor_ADMM_sc,acc_tensor_ADMM,
                 runtime_tensor_ADMM_sc,runtime_tensor_ADMM,acc_tensor_central,
                 runtime_tensor_central,adj_tensor_ADMM_sc,adj_tensor_ADMM)