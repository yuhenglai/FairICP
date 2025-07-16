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
sys.path.append(os.path.join(os.getcwd(), "../../others"))
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
    
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
import myskorch

from sklearn.utils.validation import check_is_fitted
from fairlearn.reductions._moments import ClassificationMoment

def my_pmf_predict(algo, X):
    check_is_fitted(algo)

    pred = pd.DataFrame()
    for t in range(len(algo.predictors_)):
        if algo.weights_[t] == 0:
            pred[t] = np.zeros(len(X))
        else:
            pred[t] = algo.predictors_[t].predict_proba(X)[:, 1]

    if isinstance(algo.constraints, ClassificationMoment):
        positive_probs = pred[algo.weights_.index].dot(algo.weights_).to_frame()
        return np.concatenate((1 - positive_probs, positive_probs), axis=1)
    else:
        return pred

def inner_func_test(n, dataset, dim, folder_name):
    seed = 123 * n
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, A, Y, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed, dim = dim)

    mu_val_list = [0.9, 1.2, 1.5, 2.5, 7.0, 10.0, 20.0]

    cur_mu = []
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
        
    # estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
    estimator = myskorch.FeedForwardClassifier(batch_size = 128, max_epochs = 100, module__hidden_layers = [64], module__use_batch_norm = False, module__dropout = 0, optimizer__lr = 0.001, optimizer__weight_decay = 0, optimizer__amsgrad = False, device='cpu')
    for mu_val in mu_val_list:
        exp_grad_red = ExponentiatedGradient(  estimator=estimator,
                                                constraints=EqualizedOdds(),
                                                eps = mu_val,
                                                max_iter = 50
                                            )    
        exp_grad_red.fit(X, Y, sensitive_features = A)
        # Yhat_out_test = exp_grad_red._pmf_predict(X_test)[:,1]
        # Yhat_out_cal = exp_grad_red._pmf_predict(X_cal)[:,1]

        Yhat_out_test = np.squeeze(my_pmf_predict(exp_grad_red, X_test)[:,1])
        misclassification = sum((Yhat_out_test > 0.5).astype(float) != Y_test) / Y_test.shape[0]
        print("misclassification error = " + str(misclassification))


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

        ind_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])] if A.shape[1] == 2 else [np.array([0]), np.array([1])]
        deo = 0
        yhat = (Yhat_out_test > 0.5).astype(float)
        for y_v in [0, 1]:
            y_ind = (Y_test == y_v)
            py = yhat[y_ind].sum() / yhat[y_ind].shape[0]
            for a_uni in ind_list:
                a_ind = np.all([A_test == a_uni], axis = 2).reshape(A_test.shape[0],)
                a_ind = a_ind * y_ind
                pa = yhat[a_ind].sum() / yhat[a_ind].shape[0]
                deo += np.abs(pa - py)
        cur_deo.append(deo)

        cur_mu.append(mu_val)

        df = pd.DataFrame({ 'mu_val': cur_mu,
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

    batch_size_list = [64, 128, 256]
    lr_loss_list = [0.0001, 0.001, 0.01]
    mu_val_list = [0.9, 1.2, 1.5, 2.5, 7.0, 10.0, 20.0]

    
    cur_batch_size = []
    cur_lr_loss = []
    cur_mu = []
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
        for lr_loss in lr_loss_list:

            estimator = myskorch.FeedForwardClassifier(batch_size = batch_size, max_epochs = 100, module__hidden_layers = [64], module__use_batch_norm = False, module__dropout = 0, optimizer__lr = lr_loss, optimizer__weight_decay = 0, optimizer__amsgrad = False, device='cpu')
            for mu_val in mu_val_list:
                            
                exp_grad_red = ExponentiatedGradient(  estimator=estimator,
                                                constraints=EqualizedOdds(),
                                                eps = mu_val,
                                                max_iter = 50
                                            )    
                exp_grad_red.fit(X, Y, sensitive_features = A)

                Yhat_out_test = np.squeeze(my_pmf_predict(exp_grad_red, X_test)[:,1])
                misclassification = sum((Yhat_out_test > 0.5).astype(float) != Y_test) / Y_test.shape[0]
                print("misclassification error = " + str(misclassification))


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

                ind_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])] if A.shape[1] == 2 else [np.array([0]), np.array([1])]
                deo = 0
                yhat = (Yhat_out_test > 0.5).astype(float)
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
                cur_loss.append(misclassification)

                df = pd.DataFrame({ 'batch_size': cur_batch_size,
                                    'lr_loss': cur_lr_loss,
                                    'mu_val': cur_mu,
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
    folder_name = f"./{dataset}_Reduction_{dim}dim_{mode}/"
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