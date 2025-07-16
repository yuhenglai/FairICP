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
base_path = os.getcwd() + '/data/'
sys.path.append(os.path.join(os.getcwd(), "continuous-fairness-master"))
import torch
import random
import get_dataset
import numpy as np
import multiprocessing
import matplotlib
import seaborn as sns
from multiprocessing import Process
import pandas as pd
from  FairICP import utility_functions
from  FairICP import  FairICP_learning
from scipy.stats import multivariate_normal as mvn
import warnings
warnings.filterwarnings('ignore')

os.environ['R_HOME'] = "" # path
os.environ['R_USER'] = "" # path


from rpy2.robjects.packages import importr
KPC = importr('KPC')
FOCI = importr('FOCI')
kernlab = importr('kernlab')
import rpy2.robjects
from rpy2.robjects import FloatVector

def mix_gamma(n):
    # Settings
    NumberOfMixtures = 2

    w = [0.5, 0.5]
    shapeVectors = np.array([1, 10])
    scaleVectors = np.array([1, 1])
    MeanVectors = shapeVectors * scaleVectors
    StdVectors = shapeVectors * np.square(scaleVectors)
    moments = np.square(MeanVectors) + StdVectors
    mean = np.array(w).dot(MeanVectors)
    std = np.sqrt(np.array(w).dot(moments) - np.square(mean))
    # Initialize arrays
    samples = np.zeros(n)
    # Generate samples
    for iter in range(n):
        # Get random number to select the mixture component with probability according to mixture weights
        DrawComponent = random.choices(range(NumberOfMixtures), weights=w, cum_weights=None, k=1)[0]
        # Draw sample from selected mixture component
        DrawSample = np.random.gamma(shape = shapeVectors[DrawComponent], scale = scaleVectors[DrawComponent], size = 1)

        DrawSample = (DrawSample - mean) / std
        samples[iter] = DrawSample
    return samples
    
cov = 0
def synthetic_example_md(dim_insnst = 5, dim_snst = 1, dim_noisy_a = 0, alpha = 0.5, eps = 1, n = 1000, include_A = False):
    # insensitive X
    cov_mat = np.full((dim_insnst, dim_insnst), cov)
    np.fill_diagonal(cov_mat, 1)
    X_insnst = mvn.rvs(mean = [0. for i in range(dim_insnst)], cov=cov_mat, size = n)
    if len(X_insnst.shape) == 1: X_insnst = X_insnst[:,None] 

    # sensitive X
    X_snst = np.array([[] for i in range(n)])
    A = np.array([[] for i in range(n)])
    for i in range(dim_snst):
        A_temp = mix_gamma(n)
        X_temp = np.sqrt(alpha) * A_temp + np.sqrt(1 - alpha) * np.random.randn(n)
        A = np.concatenate([A, A_temp[:,None]], axis = 1)
        X_snst = np.concatenate([X_snst, X_temp[:,None]], axis = 1)

    # additional a
    self_cor = 0
    for i in range(dim_noisy_a):
        A_temp = self_cor * A[:,-1] + np.sqrt(1 - np.square(self_cor)) * (mix_gamma(n))
        A = np.concatenate([A, A_temp[:,None]], axis = 1)

    X = np.concatenate([X_insnst, X_snst], axis = 1)
    beta = [1] * dim_insnst + [1] * dim_snst
    Y = np.dot(X, beta) + eps * np.random.randn(n)

    if include_A:
        X = np.concatenate((X, A), axis = 1)

    return X, A, Y

def known_prob(Y, A, dim_insnst = 5, alpha = 0.5, eps = 1):
    cov_mat = np.full((dim_insnst, dim_insnst), cov)
    np.fill_diagonal(cov_mat, 1)
    sig_insnst = np.ones(dim_insnst).dot(cov_mat.dot(np.ones(dim_insnst)[:,None]))

    sig_snst = A.shape[1] * (1 - alpha)

    sig2 = np.full((A.shape[0], ), sig_insnst + sig_snst + np.power(eps, 2))
    mu = np.sqrt(alpha) * np.sum(A, axis = 1)

    return - np.power(Y,2)[:,None] * (1/2/sig2)[None,:] + Y[:,None] * (mu/sig2)[None,:]

def inner_func(n, dataset, folder_name):
    seed = 123 * n
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    func = synthetic_example_md

    epochs_list = [100] 

    cur_dim_insnst = []
    cur_dim_snst = []
    cur_dim_noisy_a = []
    cur_alpha = []
    cur_eps = []
    cur_est = []
    cur_loss = []
    cur_kpcg_nn = []
    cur_pval = {"kpcg_nn": []} 

    cur_batch_size = []
    cur_lr_loss = []
    cur_lr_dis = []
    cur_mu = []
    cur_epoch = []
    
    for dim_snst in [1, 5, 10]: # [1] for setting 2
        dim_insnst = dim_snst
        for dim_noisy_a in [0]: # [1,5,10] for setting 2
            for alpha in [0.6, 0.9]:
                for eps in [np.sqrt(dim_insnst + dim_snst)]:
                    X, A, Y = func(dim_insnst = dim_insnst, dim_snst = dim_snst, dim_noisy_a = dim_noisy_a, alpha = alpha, eps = eps, n = 500)
                    X_test, A_test, Y_test = func(dim_insnst = dim_insnst, dim_snst = dim_snst, dim_noisy_a = dim_noisy_a, alpha = alpha, eps = eps, n = 400)
                    input_data_train = np.concatenate((A, X), 1)
                    input_data_test = np.concatenate((A_test, X_test), 1)
                
                    specified_density_te = known_prob(Y_test, A_test[:,:dim_snst], dim_insnst, alpha, eps)
                    y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,100,specified_density_te))
                    A_perm_index = np.argsort(y_perm_index)
                    specified_At_cpt = A_test[:,:dim_snst][A_perm_index]
                    
                    
                    for density in ["est_cpt", "est_cpt_vanilla", "est_crt", "real_cpt"]:
                        if density == "est_cpt": 
                            trained_maf = utility_functions.fit_maf([Y[:,None], A])
                            specified_density_tr = trained_maf.log_prob((np.repeat(Y, Y.shape[0]))[:,None], bijector_kwargs=utility_functions.make_bijector_kwargs(trained_maf.bijector, {'maf.': {'conditional_input': np.tile(A, (Y.shape[0], 1))}})).numpy()
                            specified_density_tr = specified_density_tr.reshape(Y.shape[0], Y.shape[0])
                        elif density == 'est_cpt_vanilla':
                            trained_maf = utility_functions.fit_maf([A, Y[:,None]])
                            specified_density_tr = trained_maf.log_prob((np.repeat(A, A.shape[0], axis = 0)), bijector_kwargs=utility_functions.make_bijector_kwargs(trained_maf.bijector, {'maf.': {'conditional_input': np.tile(Y[:,None], (Y.shape[0], 1))}})).numpy()
                            specified_density_tr = specified_density_tr.reshape(Y.shape[0], Y.shape[0])
                        elif density == "est_crt": 
                            specified_density_tr = utility_functions.fit_maf([A, Y[:,None]])
                        elif density == "real_cpt":
                            specified_density_tr = known_prob(Y, A[:,:dim_snst], dim_insnst, alpha, eps) 
                            
                        if density == 'est_cpt' or density == 'real_cpt':
                            ##############
                            y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,epochs_list[-1],specified_density_tr))
                            A_perm_index = np.argsort(y_perm_index)
                            specified_At_tr = A[A_perm_index]
                            ##############
                        elif density == 'est_cpt_vanilla':
                            ##############
                            A_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,epochs_list[-1],specified_density_tr))
                            specified_At_tr = A[A_perm_index]
                            ##############        
                        elif density == "est_crt":                                 
                            specified_At_tr = []
                            for trial in range(epochs_list[-1]):
                                test_sample = specified_density_tr.sample((A.shape[0], ), bijector_kwargs=utility_functions.make_bijector_kwargs(specified_density_tr.bijector, {'maf.': {'conditional_input': Y[:,None]}})).numpy()
                                specified_At_tr.append(test_sample)
                                
                        mu_val_density_lists = [0, 0.3, 0.5, 0.7, 0.8, 0.9]                      
                        for batch_size in [16]:
                            for lr_loss in [0.001]: 
                                for lr_dis in [0.0001]: 
                                    for mu_val in mu_val_density_lists:
                                        model =  FairICP_learning.EquiRegLearner(lr_loss = lr_loss,
                                                                                        lr_dis = lr_dis,
                                                                                    epochs = epochs_list[-1],
                                                                                    loss_steps = 1,
                                                                                    dis_steps = 1,
                                                                                    cost_pred = torch.nn.MSELoss(),
                                                                                    in_shape = X.shape[1],
                                                                                    batch_size = batch_size,
                                                                                    model_type = "linear_model",
                                                                                    lambda_vec = mu_val,
                                                                                    out_shape = 1,
                                                                                    A_shape = A.shape[1]
                                                                                    )
                                        model.fit(input_data_train, Y, check_period = epochs_list,specified_At = specified_At_tr)
                                        
                                        At_dic = {"real_cpt": specified_At_cpt}
                                        for i, cp in enumerate(model.checkpoint_list):
                                            model.model = model.cp_model_list[i]
                                            model.dis = model.cp_dis_list[i]

                                            Yhat_out_test = model.predict(input_data_test)

                                            mse_model = np.mean((Yhat_out_test-Y_test)**2)

                                            specified_At = At_dic['real_cpt']
                                            rYhat = FloatVector(Yhat_out_test)  
                                            rZ = rpy2.robjects.r.matrix(FloatVector(A_test[:,:dim_snst].T.flatten()), nrow=A_test[:,:dim_snst].shape[0], ncol=A_test[:,:dim_snst].shape[1]) # rpy2.robjects.r.matrix
                                            rY = rpy2.robjects.r.matrix(FloatVector(Y_test), nrow=A_test.shape[0], ncol=1)
        
                                            
                                            stat = KPC.KPCgraph 
                                            res_ = stat(Y = rYhat, X = rY, Z = rZ, Knn = 1)[0]
                                            res_list = np.zeros(100)
                                            for i in range(100):
                                                At_test = specified_At[i]
                                                rZt = rpy2.robjects.r.matrix(FloatVector(At_test.T.flatten()), nrow=A_test[:,:dim_snst].shape[0], ncol=A_test[:,:dim_snst].shape[1])
                                                res_list[i] = stat(Y = rYhat, X = rY, Z = rZt, Knn = 1)[0]
                                            p_val = 1.0/(100+1) * (1 + sum(res_list >= res_))
                                            cur_kpcg_nn.append(res_)
                                            cur_pval["kpcg_nn"].append(p_val)

                                            cur_dim_insnst.append(dim_insnst)
                                            cur_dim_snst.append(dim_snst)
                                            cur_dim_noisy_a.append(dim_noisy_a)
                                            cur_alpha.append(alpha)
                                            cur_eps.append(eps)
                                            cur_est.append(density)
                                            cur_loss.append(mse_model)
                                            
                                            cur_batch_size.append(batch_size)
                                            cur_lr_loss.append(lr_loss)
                                            cur_lr_dis.append(lr_dis)
                                            cur_mu.append(mu_val)
                                            cur_epoch.append(cp)
                                                
                                                
                                        df = pd.DataFrame({
                                                                'dim_insnst': cur_dim_insnst,
                                                                'dim_snst': cur_dim_snst,
                                                                'dim_noisy_a': cur_dim_noisy_a,
                                                                'alpha': cur_alpha,
                                                                'eps': cur_eps,
                                                                'density': cur_est,
                                                                'batch_size': cur_batch_size,
                                                                'lr_loss': cur_lr_loss,
                                                                'lr_dis': cur_lr_dis,
                                                                'mu_val': cur_mu,
                                                                'epochs': cur_epoch,
                                                                'loss': cur_loss,
                                                                "kpcg_nn":cur_kpcg_nn
                                                                })
                                        df.to_csv(os.path.join(folder_name, f"{dataset} {n}.csv"))


if __name__ == '__main__':
    split_num = 100
    dataset = "simulation"
    print(dir_path)
    print("dataset: " + dataset)

    orig_stdout = sys.__stdout__
    folder_name = f"./{dataset}_1/" # setting 1 or 2
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    shutil.copy(__file__, os.path.join(folder_name, os.path.basename(__file__)))
    processes = [Process(target=inner_func, args=(n, dataset, folder_name)) for n in range(split_num)]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()