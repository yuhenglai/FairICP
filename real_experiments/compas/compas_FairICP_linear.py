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
    
def inner_func_test(n, dataset, dim, folder_name):
    seed = 123 * n
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, A, Y, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed, dim = dim)
    input_data_train = np.concatenate((A, X), 1)
    input_data_test = np.concatenate((A_test, X_test), 1)

    batch_size_list = [256]
    lr_loss_list = [1e-3]
    lr_dis_list = [1e-4]
    mu_val_list = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    epochs_list = [80] 

    # utility loss
    cost_pred = torch.nn.CrossEntropyLoss()
    # base predictive model
    model_type = "linear_model"
    
    cur_batch_size = []
    cur_lr_loss = []
    cur_lr_dis = []
    cur_mu = []
    cur_epoch = []
    cur_loss = []
    cur_kpcg_nn = []
    cur_pval = {"kpcg_nn": []} 
    cur_deo = []

    specified_density_te = utility_functions.Class_density_estimation(Y_test, A_test, Y_test, A_test) 
    ##############
    y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,100,specified_density_te))
    A_perm_index = np.argsort(y_perm_index)
    specified_At = A_test[A_perm_index]
    ##############  
        
    for batch_size in batch_size_list:
        for lr_dis in lr_dis_list:
            for lr_loss in lr_loss_list:
                for mu_val in mu_val_list:
                                
                    print(f"######## batch_size{batch_size} lr_loss{lr_loss} lr_dis{lr_dis} mu_val{mu_val} begin! ########", flush = True)
                                    
                    model = FairICP_learning.EquiClassLearner(lr_loss = lr_loss,
                                                                    lr_dis = lr_dis,
                                                                epochs = epochs_list[-1],
                                                                loss_steps = 1,
                                                                dis_steps = 1 if mu_val > 0 else 0,
                                                                cost_pred = cost_pred,
                                                                in_shape = X.shape[1],
                                                                batch_size = batch_size,
                                                                model_type = model_type,
                                                                lambda_vec = mu_val,
                                                                num_classes = 2,
                                                                A_shape = A.shape[1]
                                                                )
                    model.fit(input_data_train, Y, check_period = epochs_list)
    
                    for i, cp in enumerate(model.checkpoint_list):
                        model.model = model.cp_model_list[i]
                        model.dis = model.cp_dis_list[i]

                        Yhat_out_test = model.predict(input_data_test)

                        mis_model = 1 - utility_functions.compute_acc_numpy(Yhat_out_test, Y_test)

                        rYhat = rpy2.robjects.r.matrix(FloatVector(Yhat_out_test.T.flatten()), nrow=Yhat_out_test.shape[0], ncol=Yhat_out_test.shape[1])
                        rZ = rpy2.robjects.r.matrix(FloatVector(A_test.T.flatten()), nrow=A_test.shape[0], ncol=A_test.shape[1]) # rpy2.robjects.r.matrix
                        rY = rpy2.robjects.r.matrix(FloatVector(Y_test), nrow=A_test.shape[0], ncol=1)
                        
                        stat = KPC.KPCgraph 
                        res_ = stat(Y = rYhat, X = rY, Z = rZ, Knn = 1)[0]
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
                        cur_lr_dis.append(lr_dis)
                        cur_mu.append(mu_val)
                        cur_epoch.append(cp)
                        cur_loss.append(mis_model)

                        df = pd.DataFrame({ 'batch_size': cur_batch_size,
                                            'lr_loss': cur_lr_loss,
                                            'lr_dis': cur_lr_dis,
                                            'mu_val': cur_mu,
                                            'epochs': cur_epoch,
                                            'loss': cur_loss,
                                            "kpcg_nn":cur_kpcg_nn,
                                            'pval_kpcg_nn': cur_pval["kpcg_nn"],
                                            'deo': cur_deo
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

    input_data_train = np.concatenate((A, X), 1)
    input_data_test = np.concatenate((A_test, X_test), 1)

    batch_size_list = [128, 256, 512]
    lr_loss_list = [1e-4, 1e-3, 1e-2]
    lr_dis_list = [1e-5, 1e-4, 1e-3]
    mu_val_list = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    epochs_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] 

    # utility loss
    cost_pred = torch.nn.CrossEntropyLoss()
    # base predictive model
    model_type = "linear_model"
    
    cur_batch_size = []
    cur_lr_loss = []
    cur_lr_dis = []
    cur_mu = []
    cur_epoch = []
    cur_loss = []
    cur_kpcg_nn = []
    cur_pval = {"kpcg_nn": []} 
    cur_deo = []

    specified_density_te = utility_functions.Class_density_estimation(Y_test, A_test, Y_test, A_test) 
    ##############
    y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,100,specified_density_te))
    A_perm_index = np.argsort(y_perm_index)
    specified_At = A_test[A_perm_index]
    ##############  
        
    for batch_size in batch_size_list:
        for lr_dis in lr_dis_list:
            for lr_loss in lr_loss_list:
                for mu_val in mu_val_list:
                                
                    print(f"######## batch_size{batch_size} lr_loss{lr_loss} lr_dis{lr_dis} mu_val{mu_val} begin! ########", flush = True)
                                    
                    model = FairICP_learning.EquiClassLearner(lr_loss = lr_loss,
                                                                    lr_dis = lr_dis,
                                                                epochs = epochs_list[-1],
                                                                loss_steps = 1,
                                                                dis_steps = 1 if mu_val > 0 else 0,
                                                                cost_pred = cost_pred,
                                                                in_shape = X.shape[1],
                                                                batch_size = batch_size,
                                                                model_type = model_type,
                                                                lambda_vec = mu_val,
                                                                num_classes = 2,
                                                                A_shape = A.shape[1]
                                                                )
                    model.fit(input_data_train, Y, check_period = epochs_list)
    
                    for i, cp in enumerate(model.checkpoint_list):
                        model.model = model.cp_model_list[i]
                        model.dis = model.cp_dis_list[i]

                        Yhat_out_test = model.predict(input_data_test)

                        mis_model = 1 - utility_functions.compute_acc_numpy(Yhat_out_test, Y_test)

                        rYhat = rpy2.robjects.r.matrix(FloatVector(Yhat_out_test.T.flatten()), nrow=Yhat_out_test.shape[0], ncol=Yhat_out_test.shape[1])
                        rZ = rpy2.robjects.r.matrix(FloatVector(A_test.T.flatten()), nrow=A_test.shape[0], ncol=A_test.shape[1]) # rpy2.robjects.r.matrix
                        rY = rpy2.robjects.r.matrix(FloatVector(Y_test), nrow=A_test.shape[0], ncol=1)
                        
                        stat = KPC.KPCgraph 
                        res_ = stat(Y = rYhat, X = rY, Z = rZ, Knn = 1)[0]
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
                        cur_lr_dis.append(lr_dis)
                        cur_mu.append(mu_val)
                        cur_epoch.append(cp)
                        cur_loss.append(mis_model)

                        df = pd.DataFrame({ 'batch_size': cur_batch_size,
                                            'lr_loss': cur_lr_loss,
                                            'lr_dis': cur_lr_dis,
                                            'mu_val': cur_mu,
                                            'epochs': cur_epoch,
                                            'loss': cur_loss,
                                            "kpcg_nn":cur_kpcg_nn,
                                            'pval_kpcg_nn': cur_pval["kpcg_nn"],
                                            'deo': cur_deo
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
    folder_name = f"./{dataset}_FairICP_linear_{dim}dim_{mode}/"
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