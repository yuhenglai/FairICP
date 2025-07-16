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
    

import gerryfair
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim=1, 
                 learning_rate=0.001, epochs=100, batch_size=32):
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

        self.model = nn.Sequential(
            self.layer1,
            self.relu,
            self.layer2
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.scale_factor = 0

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def fit(self, X, y):
        self.reset_parameters()
        self.scale_factor = len(y)
        # Convert to PyTorch tensors
        y = np.asarray(y).reshape(-1, 1) *  len(y)
        X = np.asarray(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, targets in dataloader:
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # if epoch % 10 == 0: print(f"Epoch {epoch}, Loss: {loss.item()}")
        return self
        
    def predict(self, X):
        # Convert to tensor and make prediction
        X = np.asarray(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.numpy() / self.scale_factor




def inner_func_test(n, dataset, dim, folder_name):
    seed = 123 * n
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X, A, Y, X_test, A_test, Y_test = get_dataset.get_train_test_data(base_path, dataset, seed, dim = dim)
    X_train = pd.DataFrame(X)
    X_prime_train = pd.DataFrame(A)
    y_train = pd.Series(Y)
    X_test = pd.DataFrame(X_test)



    batch_size_list = [128]
    lr_loss_list = [0.001]
    mu_val_list = [0.005, 0.01, 0.015, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028]

    
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
            for mu_val in mu_val_list:
                estimator = SimpleNN(input_dim = X_train.shape[1], hidden_dim = 64, epochs = 3, batch_size = batch_size, learning_rate = lr_loss)
                
                C = 100
                printflag = True
                gamma = mu_val
                fair_model = gerryfair.model.Model(C=C, printflag=printflag, gamma=gamma, fairness_def='FP', predictor=estimator)
                max_iters = 500
                fair_model.set_options(max_iters=max_iters)
                
                [errors, fp_difference] = fair_model.train(X_train, X_prime_train, y_train)


                num_classifiers = len(fair_model.classifiers)
                y_hat = None
                for c in fair_model.classifiers: 
                    last_0 = np.squeeze(c.b0.predict(X_test))
                    last_1 = np.squeeze(c.b1.predict(X_test))
                
                    new_preds = np.multiply(1.0 / num_classifiers, 1 / (1 + np.exp(-X.shape[0] * (last_0 - last_1))))
                    if y_hat is None:
                        y_hat = new_preds
                    else:
                        y_hat = np.add(y_hat, new_preds)
                Yhat_out_test_prob = y_hat
                
                Yhat_out_test = np.array(fair_model.predict(X_test))
                # np.mean((y_hat > 0.5) * 1 != Y_test)
                # Yhat_out_test = np.array(fair_model.predict(X_test))
                misclassification = sum((Yhat_out_test > 0.5).astype(float) != Y_test) / Y_test.shape[0]
                print("misclassification error = " + str(misclassification))
                            
                rYhat = FloatVector(Yhat_out_test_prob)
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

    X_train = pd.DataFrame(X)
    X_prime_train = pd.DataFrame(A)
    y_train = pd.Series(Y)
    X_test = pd.DataFrame(X_test)

    batch_size_list = [128, 256]
    lr_loss_list = [0.001, 0.005]
    mu_val_list = [0.005, 0.01, 0.015, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028]

    
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
            for mu_val in mu_val_list:
                estimator = SimpleNN(input_dim = X_train.shape[1], hidden_dim = 64, epochs = 3, batch_size = batch_size, learning_rate = lr_loss)
                
                C = 100
                printflag = True
                gamma = mu_val
                fair_model = gerryfair.model.Model(C=C, printflag=printflag, gamma=gamma, fairness_def='FP', predictor=estimator)
                max_iters = 500
                fair_model.set_options(max_iters=max_iters)
                
                [errors, fp_difference] = fair_model.train(X_train, X_prime_train, y_train)


                num_classifiers = len(fair_model.classifiers)
                y_hat = None
                for c in fair_model.classifiers: 
                    last_0 = np.squeeze(c.b0.predict(X_test))
                    last_1 = np.squeeze(c.b1.predict(X_test))
                
                    new_preds = np.multiply(1.0 / num_classifiers, 1 / (1 + np.exp(-X.shape[0] * (last_0 - last_1))))
                    if y_hat is None:
                        y_hat = new_preds
                    else:
                        y_hat = np.add(y_hat, new_preds)
                Yhat_out_test_prob = y_hat
                
                Yhat_out_test = np.array(fair_model.predict(X_test))
                # np.mean((y_hat > 0.5) * 1 != Y_test)
                # Yhat_out_test = np.array(fair_model.predict(X_test))
                misclassification = sum((Yhat_out_test > 0.5).astype(float) != Y_test) / Y_test.shape[0]
                print("misclassification error = " + str(misclassification))
                            
                rYhat = FloatVector(Yhat_out_test_prob)
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
    folder_name = f"./{dataset}_gerryfair_{dim}dim_{mode}/"
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