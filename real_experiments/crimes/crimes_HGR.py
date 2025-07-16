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


def chi_squared_l1_kde(X, Y, Z):
    return torch.mean(chi_2_cond(X, Y, Z, kde))
class NetRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NetRegression, self).__init__()
        size = 64
        self.first = nn.Linear(input_size, size)
        self.last = nn.Linear(size, num_classes)
        
        # self.last = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = F.relu(self.first(x))
        out = self.last(out)
        return out

def regularized_learning(x_train, y_train, z_train, model, fairness_penalty, bs = 64, lr=1e-5, num_epochs=[10], penalty=1.0):
    # wrap dataset in torch tensors
    Y = torch.tensor(y_train.astype(np.float32))
    X = torch.tensor(x_train.astype(np.float32))
    Z = torch.tensor(z_train.astype(np.float32))
    dataset = data_utils.TensorDataset(X, Y, Z)
    dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=bs, shuffle=True)

    # mse regression objective
    data_fitting_loss = nn.MSELoss()

    # stochastic optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_list = []
    for j in range(num_epochs[-1]):
        for i, (x, y, z) in enumerate(dataset_loader):
            def closure():
                optimizer.zero_grad()
                outputs = model(x).flatten()
                loss = data_fitting_loss(outputs, y)
                loss += penalty*fairness_penalty(outputs, z, y)
                loss.backward()
                return loss

            optimizer.step(closure)
        if j + 1 in num_epochs: model_list.append(copy.deepcopy(model))
    return model_list

def inner_func_test(n, dataset, dim, folder_name):
    seed = 123 * n
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, A, Y, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed, dim = dim)

    batch_size_list = [64]
    lr_loss_list = [0.001]
    mu_val_list = [0, 0.25, 0.5, 1, 2, 4, 8, 16]
    epochs_list = [40] 
    
    cur_batch_size = []
    cur_lr_loss = []
    cur_mu = []
    cur_epoch = []
    cur_loss = []
    cur_kpcg_nn = []
    cur_pval = {"kpcg_nn": []} 

    specified_density_te = utility_functions.MAF_density_estimation(Y_test, A_test, Y_test, A_test) 
    ##############
    y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,100,specified_density_te))
    A_perm_index = np.argsort(y_perm_index)
    specified_At = A_test[A_perm_index]
    ##############  
        
    for batch_size in batch_size_list:
        for lr_loss in lr_loss_list:
            for mu_val in mu_val_list:
                model = NetRegression(X.shape[1], 1)

                lr = lr_loss

                # $\chi^2|_1$
                penalty_coefficient = mu_val
                penalty = chi_squared_l1_kde

                model_list = regularized_learning(X, Y, A, model=model, fairness_penalty=penalty, bs = batch_size, lr=lr, num_epochs=epochs_list, penalty = penalty_coefficient)       
                

                for i, cp in enumerate(epochs_list):
                    model = model_list[i]

                    Yhat_out_test = model(torch.tensor(X_test.astype(np.float32))).detach().flatten().numpy()

                    mse_model = np.mean((Yhat_out_test-Y_test)**2)

                    rYhat = FloatVector(Yhat_out_test)  
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

                    cur_batch_size.append(batch_size)
                    cur_lr_loss.append(lr_loss)
                    cur_mu.append(mu_val)
                    cur_epoch.append(cp)
                    cur_loss.append(mse_model)

                    df = pd.DataFrame({ 'batch_size': cur_batch_size,
                                        'lr_loss': cur_lr_loss,
                                        'mu_val': cur_mu,
                                        'epochs': cur_epoch,
                                        'loss': cur_loss,
                                        "kpcg_nn":cur_kpcg_nn,
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

    batch_size_list = [16, 32, 64]
    lr_loss_list = [1e-4, 1e-3, 1e-2]
    mu_val_list = [0, 0.25, 0.5, 1, 2, 4, 8, 16]
    epochs_list = [20, 40, 60, 80, 100] 

    
    cur_batch_size = []
    cur_lr_loss = []
    cur_mu = []
    cur_epoch = []
    cur_loss = []
    cur_kpcg_nn = []
    cur_pval = {"kpcg_nn": []} 

    specified_density_te = utility_functions.MAF_density_estimation(Y_test, A_test, Y_test, A_test) 
    ##############
    y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,100,specified_density_te))
    A_perm_index = np.argsort(y_perm_index)
    specified_At = A_test[A_perm_index]
    ##############  
        
    for batch_size in batch_size_list:
        for lr_loss in lr_loss_list:
            for mu_val in mu_val_list:
                model = NetRegression(X.shape[1], 1)

                lr = lr_loss

                # $\chi^2|_1$
                penalty_coefficient = mu_val
                penalty = chi_squared_l1_kde

                model_list = regularized_learning(X, Y, A, model=model, fairness_penalty=penalty, bs = batch_size, lr=lr, num_epochs=epochs_list, penalty = penalty_coefficient)       
                

                for i, cp in enumerate(epochs_list):
                    model = model_list[i]

                    Yhat_out_test = model(torch.tensor(X_test.astype(np.float32))).detach().flatten().numpy()

                    mse_model = np.mean((Yhat_out_test-Y_test)**2)

                    rYhat = FloatVector(Yhat_out_test)  
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

                    cur_batch_size.append(batch_size)
                    cur_lr_loss.append(lr_loss)
                    cur_mu.append(mu_val)
                    cur_epoch.append(cp)
                    cur_loss.append(mse_model)

                    df = pd.DataFrame({ 'batch_size': cur_batch_size,
                                        'lr_loss': cur_lr_loss,
                                        'mu_val': cur_mu,
                                        'epochs': cur_epoch,
                                        'loss': cur_loss,
                                        "kpcg_nn":cur_kpcg_nn,
                                        'pval_kpcg_nn': cur_pval["kpcg_nn"]
                                        })
                    df.to_csv(os.path.join(folder_name, f"{dataset}_fold_{fold_index}_results.csv"))

if __name__ == '__main__':
    dataset = "crimes"
    dim = 3 # or 1
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