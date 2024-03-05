import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import tensorflow as tf
tf.autograph.set_verbosity(0)
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
import tensorflow_probability as tfp
import re
tfd = tfp.distributions
tfb = tfp.bijectors

#####################################################
##################### MAF ###########################
#####################################################
def make_bijector_kwargs(bijector, name_to_kwargs):

    if hasattr(bijector, 'bijectors'):
        return {b.name: make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    else:
        for name_regex, kwargs in name_to_kwargs.items():
            if re.match(name_regex, bijector.name):
                return kwargs
    return {}

def fit_maf(data, num_mades = 5, hidden_units = [64, 64], activation = 'relu', n_epochs = 1000, n_disp = 1000, patience = 50): # data is a list, data[1]: conditions(Y), data[0]: X
    base_distribution = tfd.Normal(loc=0., scale=1.)

    bijectors=[]
    event_shape = data[0].shape[1]
    cond_shape = data[1].shape[1]


    for i in range(num_mades):
        made = tfb.AutoregressiveNetwork(params=2, event_shape=(event_shape,), hidden_units=hidden_units, conditional=True, conditional_event_shape=(cond_shape,), activation=activation, kernel_initializer = tf.keras.initializers.RandomUniform(minval=-1e-5, maxval=1e-5, seed=None)) 
        masked_auto_i = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made, name = "maf" + str(i))
        bijectors.append(masked_auto_i)

    flow_bijector = tfb.Chain(list(reversed(bijectors)))
    maf = tfd.TransformedDistribution(tfd.Sample(base_distribution, sample_shape=[event_shape]), flow_bijector,)

    x_ = Input(shape=(event_shape,), dtype=tf.float32)
    c_ = Input(shape=(cond_shape,), dtype=tf.float32)
    log_prob_ = maf.log_prob(x_, bijector_kwargs=make_bijector_kwargs(maf.bijector, {'maf.': {'conditional_input': c_}}))
    model = Model([x_,c_], log_prob_)

    ns = data[0].shape[0]
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=lambda _, log_prob: -log_prob)

    batch_size = int(ns/10) 
    
    es = EarlyStopping(monitor = 'val_loss', patience = patience, verbose = 1, min_delta = 1e-6)
    epoch_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: False)    

    y_empty = np.empty((ns, 0), dtype=np.float32)

    model.fit(  x=data,
                y=y_empty,
                batch_size=batch_size,
                epochs=n_epochs,
                validation_split=0.2,
                shuffle=True,
                verbose=False,
                callbacks=[epoch_callback, es]
                )
    return maf

#####################################################
################# Density Estimation ################
#####################################################

def MAF_density_estimation(est_on_Y, est_on_A, Y, A):
    trained_maf = fit_maf([est_on_Y[:,None], est_on_A])
    if Y.shape[0] > 2000:
        log_lik_mat = np.array([])
        for Y_temp in np.split(Y, np.arange(2000, Y.shape[0], 2000)):
            log_lik_temp = trained_maf.log_prob((np.repeat(Y_temp, Y.shape[0]))[:,None], bijector_kwargs=make_bijector_kwargs(trained_maf.bijector, {'maf.': {'conditional_input': np.tile(A, (Y_temp.shape[0], 1))}})).numpy()
            log_lik_mat = np.concatenate([log_lik_mat, log_lik_temp])
    else:
        log_lik_mat = trained_maf.log_prob((np.repeat(Y, Y.shape[0]))[:,None], bijector_kwargs=make_bijector_kwargs(trained_maf.bijector, {'maf.': {'conditional_input': np.tile(A, (Y.shape[0], 1))}})).numpy()
    log_lik_mat = log_lik_mat.reshape(Y.shape[0], Y.shape[0])
    return log_lik_mat

#####################################################
####################### CPT #########################
#####################################################

# generate CPT copies of X in general case
# log_lik_mat[i,j] = q(X[i]|Z[j]) where q(x|z) is the conditional density for X|Z
def generate_X_CPT(nstep,M,log_lik_mat,Pi_init=[]):
    n = log_lik_mat.shape[0]
    if len(Pi_init)==0:
        Pi_init = np.arange(n,dtype=int)
    Pi_ = generate_X_CPT_MC(nstep,log_lik_mat,Pi_init)
    Pi_mat = np.zeros((M,n),dtype=int)
    for m in range(M):
        Pi_mat[m] = generate_X_CPT_MC(nstep,log_lik_mat,Pi_)
    return Pi_mat

def generate_X_CPT_MC(nstep,log_lik_mat,Pi):
    n = len(Pi)
    npair = np.floor(n/2).astype(int)
    for istep in range(nstep):
        perm = np.random.choice(n,n,replace=False)
        inds_i = perm[0:npair]
        inds_j = perm[npair:(2*npair)]
        # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
        log_odds = log_lik_mat[Pi[inds_i],inds_j] + log_lik_mat[Pi[inds_j],inds_i] \
            - log_lik_mat[Pi[inds_i],inds_i] - log_lik_mat[Pi[inds_j],inds_j]
        # log_odds[np.isnan(log_odds)] = -np.Inf
        swaps = np.random.binomial(1,1/(1+np.exp(-np.maximum(-500,log_odds))))
        Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps*(Pi[inds_j]-Pi[inds_i]), Pi[inds_j] - \
            swaps*(Pi[inds_j]-Pi[inds_i])   
    return Pi

#####################################################
###################### Model ########################
#####################################################

# defining discriminator class (for regression)
class reg_discriminator(nn.Module):

    def __init__(self, inp, out=1, n_hidden = 64):

        super(reg_discriminator, self).__init__()
        self.net = nn.Sequential(
                                 nn.Linear(inp,n_hidden),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_hidden, n_hidden),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_hidden, n_hidden),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(n_hidden,out),
                                 nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.net(x)
        return x

# defining discriminator class (for classification)
class class_discriminator(nn.Module):

    def __init__(self, inp, out=1, n_hidden = 64):

        super(class_discriminator, self).__init__()
        self.net = nn.Sequential(
                                    nn.Linear(inp,n_hidden),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(n_hidden, n_hidden),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(n_hidden, n_hidden),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(n_hidden,out),
                                    nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.net(x)
        return x

# Define linear model
class linear_model(torch.nn.Module):
    def __init__(self,
                 in_shape=1,
                 out_shape=2):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.build_model()

    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.out_shape, bias=True),
        )

    def forward(self, x):
        return torch.squeeze(self.base_model(x))

# Define deep model for regression
class deep_reg_model(torch.nn.Module):
    def __init__(self,
                 in_shape=1,
                 out_shape=1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64 #in_shape*10
        self.out_shape = out_shape
        self.build_model()

    def build_model(self):
        self.base_model = nn.Sequential(
                nn.Linear(self.in_shape, self.dim_h, bias=True),
                nn.ReLU(),
                nn.Linear(self.dim_h, self.out_shape, bias=True),
        )

    def forward(self, x):
        return torch.squeeze(self.base_model(x))
    
# Define deep model for classification
class deep_class_model(torch.nn.Module):
    def __init__(self,
                 in_shape=1,
                 out_shape=1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64 #in_shape*10
        self.out_shape = out_shape
        self.build_model()

    def build_model(self):
        self.base_model = nn.Sequential(
                nn.Linear(self.in_shape, self.dim_h, bias=True),
                nn.ReLU(),
                nn.Linear(self.dim_h, self.out_shape, bias=True),
                # nn.Sigmoid(),
        )

    def forward(self, x):
        return torch.squeeze(self.base_model(x))

class pandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(pandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()    

#######################################################
############## Multiclass Y Density Estimation ########
#######################################################

def Class_density_estimation(est_on_Y, est_on_A, Y, A):
    num_classes = len(np.unique(est_on_Y))
    model = deep_class_model(in_shape = A.shape[1], out_shape = num_classes)
    clf_optimizer = torch.optim.Adam(model.parameters())
    scaler = StandardScaler()
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),
                                                columns=df.columns,
                                                index=df.index)
    epochs = 200
    batch_size = 32
    A_train = pd.DataFrame(data=est_on_A)
    scaler.fit(A_train)
    A_train = A_train.pipe(scale_df, scaler)
    Y_train = pd.DataFrame(data=est_on_Y)
    train_data = pandasDataSet(A_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # pred_loss = nn.BCELoss(reduce=True)
    pred_loss = torch.nn.CrossEntropyLoss()
    for i in range(epochs):
        for a, y in train_loader:
            Yhat = model(a)
            model.zero_grad()
            loss = pred_loss(Yhat, y.squeeze().long())
            loss.backward()
            clf_optimizer.step()

    log_lik_mat = np.zeros([Y.shape[0], Y.shape[0]])
    A = pd.DataFrame(data=A)
    A = A.pipe(scale_df, scaler)
    test_data = pandasDataSet(A)
    sm = torch.nn.Softmax()
    for j in range(Y.shape[0]):
        with torch.no_grad():
            Yhat = model(test_data.tensors[0][j]) 
        Yhat = sm(Yhat).numpy() 
        for k in range(num_classes): log_lik_mat[Y == k, j] = np.log(Yhat[k])

    return log_lik_mat    
    

def calc_accuracy(outputs,Y): #Care outputs are going to be in dimension 2
    max_vals, max_indices = torch.max(outputs,1)
    acc = (max_indices == Y).sum().detach().cpu().numpy()/max_indices.size()[0]
    return acc

def compute_acc(Yhat,Y):
    _, predicted = torch.max(Yhat, 1)
    total = Y.size(0)
    correct = (predicted == Y).sum().item()
    acc = correct/total
    return acc

def compute_acc_numpy(Yhat,Y):
    Yhat = torch.from_numpy(Yhat)
    Y = torch.from_numpy(Y)

    return compute_acc(Yhat,Y)

def pytorch_standard_scaler(x):
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x