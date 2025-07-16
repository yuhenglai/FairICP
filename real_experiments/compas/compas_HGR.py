import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import shutil
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import importlib
base_path = os.path.join(os.getcwd(), '../../data/')
sys.path.append(os.path.join(os.getcwd(), "../.."))
sys.path.append(os.path.join(os.getcwd(), "../../others/continuous-fairness-master"))
import torch
import random
import get_dataset
import numpy as np
import multiprocessing
import matplotlib
import seaborn as sns
import copy
from multiprocessing import Process
import pandas as pd
from FairICP import utility_functions
from FairICP import FairICP_learning
from sklearn.model_selection import KFold
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions
from torch.nn.parameter import Parameter
import torch.utils.data as data_utils
from collections import namedtuple
warnings.filterwarnings('ignore')

os.environ['R_HOME'] = "" # your path
os.environ['R_USER'] = "" # your path
from rpy2.robjects.packages import importr
KPC = importr('KPC')
kernlab = importr('kernlab')
import rpy2.robjects
from rpy2.robjects import FloatVector
    

from facl.independence.density_estimation.pytorch_kde import kde
from facl.independence.hgr import chi_2_cond, hgr_cond
class NetRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NetRegression, self).__init__()
        size = 64
        self.first = nn.Linear(input_size, size)
        self.last = nn.Linear(size, num_classes)       
    
    def forward(self, x):
        out = F.selu( self.first(x) )
        out = self.last(out)
        return out

# class NetRegression(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NetRegression, self).__init__()
#         self.last = nn.Linear(input_size, num_classes)       
    
#     def forward(self, x):
#         out = self.last(x)
#         return out
    

def EntropyToProba(entropy): #Only for X Tensor of dimension 2
    return entropy[:,1].exp() / entropy.exp().sum(dim=1)

def calc_accuracy(outputs,Y): #Care outputs are going to be in dimension 2
    max_vals, max_indices = torch.max(outputs,1)
    acc = (max_indices == Y).sum().numpy()/max_indices.size()[0]
    return acc

def inner_func_test(n, dataset, dim, folder_name):
    seed = 123 * n
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, A, Y, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed, dim = dim)

    batch_size_list = [128]
    lr_loss_list = [0.001]
    mu_val_list = [0, 0.0375, 0.075, 0.125, 0.25, 0.5, 1]
    epochs_list = [20]

    
    cur_batch_size = []
    cur_lr_loss = []
    cur_mu = []
    cur_epoch = []
    cur_loss = []
    cur_kpcg_nn = []
    cur_deo = []
    cur_pval = {"kpcg_nn": []} 

    specified_density_te = utility_functions.Class_density_estimation(Y_test, A_test, Y_test, A_test) 
    ##############
    y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,100,specified_density_te))
    A_perm_index = np.argsort(y_perm_index)
    specified_At = A_test[A_perm_index]
    ##############  
        
    for batch_size in batch_size_list:
        for lr_loss in lr_loss_list:
            for mu_val in mu_val_list:
                            
                # Hyper Parameters 
                input_size = X.shape[1]
                num_classes = 2
                num_epochs = epochs_list[-1]
                batchRenyi = 256
                learning_rate = lr_loss
                lambda_renyi = mu_val 
                model_list = []

                cfg_factory=namedtuple('Config', 'model  batch_size num_epochs lambda_renyi batchRenyi learning_rate input_size num_classes' )
                config = cfg_factory(NetRegression, batch_size, num_epochs, lambda_renyi, batchRenyi, learning_rate, input_size, num_classes)

                model = config.model(config.input_size, config.num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0)

                train_target = torch.tensor(Y.astype(np.longlong)).long()
                train_data = torch.tensor(X.astype(np.float32))
                train_protect = torch.tensor(A).float()
                train_tensor = data_utils.TensorDataset(train_data, train_target)

                for epoch in range(config.num_epochs):
                    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = config.batch_size, shuffle = True)
                    for i, (x, y) in enumerate(train_loader):
                        optimizer.zero_grad()
                        outputs = model(x)
                        #Select a renyi regularization mini batch and compute the value of the model on it
                        frac=config.batchRenyi/train_data.shape[0]
                        foo = torch.bernoulli(frac*torch.ones(train_data.shape[0])).byte()
                        br = train_data[foo, : ]
                        pr = train_protect[foo, :]
                        yr = train_target[foo].float()
                        ren_outs = model(br)
                    
                        #Compute the usual loss of the prediction
                        loss =  criterion(outputs, y)
                        
                        #Compte the fairness penalty for positive labels only since we optimize for DEO
                        delta =  EntropyToProba(ren_outs)
                        #r2 = chi_squared_kde( delta, pr[yr==1.])
                        r2 = chi_2_cond(delta, pr, yr, kde, mode = "sum") 
                        
                        loss += config.lambda_renyi * r2
                        
                        #In Adam we trust
                        loss.backward()
                        optimizer.step()
                    if epoch + 1 in epochs_list: model_list.append(copy.deepcopy(model))

                for i, cp in enumerate(epochs_list):
                    model = model_list[i]
                    Yhat_out_test = EntropyToProba(model(torch.tensor(X_test.astype(np.float32)))).detach().flatten().numpy()
                    
                    Yhat_out_test = np.concatenate([(1 - Yhat_out_test)[:,None], Yhat_out_test[:,None]], axis = 1)
                    mis_model = 1 - utility_functions.compute_acc_numpy(Yhat_out_test, Y_test)
                    
                    rYhat = rpy2.robjects.r.matrix(FloatVector(Yhat_out_test.T.flatten()), nrow=Yhat_out_test.shape[0], ncol=Yhat_out_test.shape[1])
                    rZ = rpy2.robjects.r.matrix(FloatVector(A_test.T.flatten()), nrow=A_test.shape[0], ncol=A_test.shape[1]) # rpy2.robjects.r.matrix
                    rY = rpy2.robjects.r.matrix(FloatVector(Y_test), nrow=A_test.shape[0], ncol=1)
                    
                    stat = KPC.KPCgraph 
                    res_ = stat(Y = rYhat, X = rY, Z = rZ, Knn = 1)[0]
                    print(res_)
                    res_list = np.zeros(100)
                    for j in range(100):
                        At_test = specified_At[j]
                        rZt = rpy2.robjects.r.matrix(FloatVector(At_test.T.flatten()), nrow=A_test.shape[0], ncol=A_test.shape[1])
                        res_list[j] = stat(Y = rYhat, X = rY, Z = rZt, Knn = 1)[0]
                    p_val = 1.0/(100+1) * (1 + sum(res_list >= res_))
                    cur_kpcg_nn.append(res_)
                    cur_pval["kpcg_nn"].append(p_val)

                    ind_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])] if A.shape[1] == 2 else [np.array([0]), np.array([1])]
                    deo = 0
                    yhat = (Yhat_out_test[:,1] > 0.5).astype(float)
                    for y_v in [0, 1]:
                        y_ind = (Y_test == y_v)
                        py = yhat[y_ind].sum() / yhat[y_ind].shape[0]
                        for a_uni in ind_list:
                            a_ind = np.all([A_test == a_uni], axis = 2).reshape(A_test.shape[0],)
                            a_ind = a_ind * y_ind
                            pa = yhat[a_ind].sum() / yhat[a_ind].shape[0]
                            deo += np.abs(pa - py)
                    cur_deo.append(deo)

                    cur_batch_size.append(batch_size)
                    cur_lr_loss.append(lr_loss)
                    cur_mu.append(mu_val)
                    cur_epoch.append(cp)
                    cur_loss.append(mis_model)

                    df = pd.DataFrame({ 'batch_size': cur_batch_size,
                                        'lr_loss': cur_lr_loss,
                                        'mu_val': cur_mu,
                                        'epochs': cur_epoch,
                                        'loss': cur_loss,
                                        "kpcg_nn":cur_kpcg_nn,
                                        'deo': cur_deo,
                                        'pval_kpcg_nn': cur_pval["kpcg_nn"]
                                        })
                    df.to_csv(os.path.join(folder_name, f"{dataset}_test_{n}_results.csv"))

def inner_func_validate(fold_index, train_indices, test_indices, X_, A_, Y_, dataset, dim, folder_name):
    seed = 123 + fold_index  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Get train and test data for the current fold
    X = X_[train_indices]
    A = A_[train_indices]
    Y = Y_[train_indices]

    X_test = X_[test_indices]
    A_test = A_[test_indices]
    Y_test = Y_[test_indices]

    batch_size_list = [128, 256, 512]
    lr_loss_list = [0.0001, 0.001, 0.01]
    mu_val_list = [0, 0.0375, 0.075, 0.125, 0.25, 0.5, 1]
    epochs_list = [20, 40, 60, 80, 100]

    
    cur_batch_size = []
    cur_lr_loss = []
    cur_mu = []
    cur_epoch = []
    cur_loss = []
    cur_kpcg_nn = []
    cur_deo = []
    cur_pval = {"kpcg_nn": []} 

    specified_density_te = utility_functions.Class_density_estimation(Y_test, A_test, Y_test, A_test) 
    ##############
    y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,100,specified_density_te))
    A_perm_index = np.argsort(y_perm_index)
    specified_At = A_test[A_perm_index]
    ##############  
        
    for batch_size in batch_size_list:
        for lr_loss in lr_loss_list:
            for mu_val in mu_val_list:
                            
                # Hyper Parameters 
                input_size = X.shape[1]
                num_classes = 2
                num_epochs = epochs_list[-1]
                batchRenyi = 256
                learning_rate = lr_loss
                lambda_renyi = mu_val 
                model_list = []

                cfg_factory=namedtuple('Config', 'model  batch_size num_epochs lambda_renyi batchRenyi learning_rate input_size num_classes' )
                config = cfg_factory(NetRegression, batch_size, num_epochs, lambda_renyi, batchRenyi, learning_rate, input_size, num_classes)

                model = config.model(config.input_size, config.num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0)

                train_target = torch.tensor(Y.astype(np.longlong)).long()
                train_data = torch.tensor(X.astype(np.float32))
                train_protect = torch.tensor(A).float()
                train_tensor = data_utils.TensorDataset(train_data, train_target)

                for epoch in range(config.num_epochs):
                    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = config.batch_size, shuffle = True)
                    for i, (x, y) in enumerate(train_loader):
                        optimizer.zero_grad()
                        outputs = model(x)
                        #Select a renyi regularization mini batch and compute the value of the model on it
                        frac=config.batchRenyi/train_data.shape[0]
                        foo = torch.bernoulli(frac*torch.ones(train_data.shape[0])).byte()
                        br = train_data[foo, : ]
                        pr = train_protect[foo, :]
                        yr = train_target[foo].float()
                        ren_outs = model(br)
                    
                        #Compute the usual loss of the prediction
                        loss =  criterion(outputs, y)
                        
                        #Compte the fairness penalty for positive labels only since we optimize for DEO
                        delta =  EntropyToProba(ren_outs)
                        #r2 = chi_squared_kde( delta, pr[yr==1.])
                        r2 = chi_2_cond(delta, pr, yr, kde, mode = "sum") 
                        
                        loss += config.lambda_renyi * r2
                        
                        #In Adam we trust
                        loss.backward()
                        optimizer.step()
                    if epoch + 1 in epochs_list: model_list.append(copy.deepcopy(model))

                for i, cp in enumerate(epochs_list):
                    model = model_list[i]
                    Yhat_out_test = EntropyToProba(model(torch.tensor(X_test.astype(np.float32)))).detach().flatten().numpy()
                    
                    Yhat_out_test = np.concatenate([(1 - Yhat_out_test)[:,None], Yhat_out_test[:,None]], axis = 1)
                    mis_model = 1 - utility_functions.compute_acc_numpy(Yhat_out_test, Y_test)
                    
                    rYhat = rpy2.robjects.r.matrix(FloatVector(Yhat_out_test.T.flatten()), nrow=Yhat_out_test.shape[0], ncol=Yhat_out_test.shape[1])
                    rZ = rpy2.robjects.r.matrix(FloatVector(A_test.T.flatten()), nrow=A_test.shape[0], ncol=A_test.shape[1]) # rpy2.robjects.r.matrix
                    rY = rpy2.robjects.r.matrix(FloatVector(Y_test), nrow=A_test.shape[0], ncol=1)
                    
                    stat = KPC.KPCgraph 
                    res_ = stat(Y = rYhat, X = rY, Z = rZ, Knn = 1)[0]
                    print(res_)
                    res_list = np.zeros(100)
                    for j in range(100):
                        At_test = specified_At[j]
                        rZt = rpy2.robjects.r.matrix(FloatVector(At_test.T.flatten()), nrow=A_test.shape[0], ncol=A_test.shape[1])
                        res_list[j] = stat(Y = rYhat, X = rY, Z = rZt, Knn = 1)[0]
                    p_val = 1.0/(100+1) * (1 + sum(res_list >= res_))
                    cur_kpcg_nn.append(res_)
                    cur_pval["kpcg_nn"].append(p_val)

                    ind_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])] if A.shape[1] == 2 else [np.array([0]), np.array([1])]
                    deo = 0
                    yhat = (Yhat_out_test[:,1] > 0.5).astype(float)
                    for y_v in [0, 1]:
                        y_ind = (Y_test == y_v)
                        py = yhat[y_ind].sum() / yhat[y_ind].shape[0]
                        for a_uni in ind_list:
                            a_ind = np.all([A_test == a_uni], axis = 2).reshape(A_test.shape[0],)
                            a_ind = a_ind * y_ind
                            pa = yhat[a_ind].sum() / yhat[a_ind].shape[0]
                            deo += np.abs(pa - py)
                    cur_deo.append(deo)

                    cur_batch_size.append(batch_size)
                    cur_lr_loss.append(lr_loss)
                    cur_mu.append(mu_val)
                    cur_epoch.append(cp)
                    cur_loss.append(mis_model)

                    df = pd.DataFrame({ 'batch_size': cur_batch_size,
                                        'lr_loss': cur_lr_loss,
                                        'mu_val': cur_mu,
                                        'epochs': cur_epoch,
                                        'loss': cur_loss,
                                        "kpcg_nn":cur_kpcg_nn,
                                        'deo': cur_deo,
                                        'pval_kpcg_nn': cur_pval["kpcg_nn"]
                                        })
                    df.to_csv(os.path.join(folder_name, f"{dataset}_fold_{fold_index}_results.csv"))

if __name__ == '__main__':
    dataset = "compas"
    dim = 2
    mode = str(sys.argv[1])
    print(dir_path)
    print("dataset: " + dataset)
    split_num = 100 if mode == 'test' else 10 # test or validate

    orig_stdout = sys.__stdout__
    folder_name = f"./{dataset}_HGR_{dim}dim_{mode}/"
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    shutil.copy(__file__, os.path.join(folder_name, os.path.basename(__file__)))

    if mode == 'val':
        X_, A_, Y_ = get_dataset.get_full_dataset(base_path, dataset, dim=dim)
        kf = KFold(n_splits=split_num, shuffle=True, random_state=123)
        splits = list(kf.split(X_))

        processes = []
        for fold_index, (train_indices, test_indices) in enumerate(splits):
            p = Process(target=inner_func_validate, args=(fold_index, train_indices, test_indices, X_, A_, Y_, dataset, dim, folder_name))
            processes.append(p)
    elif mode == 'test':
        processes = [Process(target=inner_func_test, args=(n, dataset, dim, folder_name)) for n in range(split_num)]
    
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()