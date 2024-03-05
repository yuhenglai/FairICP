
import numpy as np
import pandas as pd
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from FairICP import utility_functions
from FairICP.utility_functions import linear_model
from FairICP.utility_functions import deep_class_model, deep_reg_model, reg_discriminator, class_discriminator

class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


def pretrain_adversary_fast_loader(dis, model, x, y, a, at, optimizer, criterion, lambdas):
    yhat = model(x).detach()
    dis.zero_grad()
    if len(yhat.size())==1:
        yhat = yhat.unsqueeze(dim=1)
    real = torch.cat((yhat,at,y),1)
    fake = torch.cat((yhat,a,y),1)
    in_dis = torch.cat((real, fake), 0)
    out_dis = dis(in_dis)
    labels = torch.cat((torch.ones(real.shape[0],1), torch.zeros(fake.shape[0],1)), 0)
    loss = (criterion(out_dis, labels) * lambdas).mean()
    loss.backward()
    optimizer.step()
    return dis

def pretrain_adversary(dis, model, data_loader, optimizer, criterion, lambdas):
    for x, y, a, at in data_loader:
        dis = pretrain_adversary_fast_loader(dis,
                                             model,
                                             x,
                                             y,
                                             a,
                                             at,
                                             optimizer,
                                             criterion,
                                             lambdas)
    return dis

##############################################################
##################### Classification Part ####################
##############################################################

def train_classifier(model, dis, data_loader, pred_loss, dis_loss,
                     clf_optimizer, adv_optimizer, lambdas,
                     dis_steps, loss_steps, num_classes):

    # Train adversary
    for i in range(dis_steps):
        for x, y, a, at in data_loader:
            yhat = model(x).detach()
            dis.zero_grad()
            if len(yhat.size())==1:
                yhat = yhat.unsqueeze(dim=1)
            real = torch.cat((yhat,at,y),1)
            fake = torch.cat((yhat,a,y),1)
            in_dis = torch.cat((real, fake), 0)
            out_dis = dis(in_dis)
            labels = torch.cat((torch.ones(real.shape[0],1), torch.zeros(fake.shape[0],1)), 0)
            loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
            loss_adv.backward()
            adv_optimizer.step()

    # Train predictor
    for i in range(loss_steps):
        for x, y, a, at in data_loader:
            yhat = model(x)
            if len(yhat.size())==1:
                yhat = yhat.unsqueeze(dim=1)

            # y_one_hot = torch.zeros(len(y), num_classes).scatter_(1, y.long(), 1.)
            fake = torch.cat((yhat,a,y),1)
            real = torch.cat((yhat,at,y),1)

            # loss_second_moment = covariance_diff_biased(fake, real)

            in_dis = torch.cat((real, fake), 0)
            model.zero_grad()
            out_dis = dis(in_dis)
            labels = torch.cat((torch.zeros(real.shape[0],1), torch.ones(fake.shape[0],1)), 0)
            clf_loss = (1.0-lambdas)*pred_loss(yhat, y.squeeze().long())
            clf_loss += (dis_loss(out_dis, labels) * lambdas).mean()
            clf_loss.backward()
            clf_optimizer.step()

            break

    return model, dis

class EquiClassLearner:

    def __init__(self,
                 lr_loss,
                 lr_dis,
                 epochs,
                 loss_steps,
                 dis_steps,
                 cost_pred,
                 in_shape,
                 batch_size,
                 model_type,
                 lambda_vec,
                 num_classes,
                 A_shape):

        self.lr_loss = lr_loss
        self.lr_dis = lr_dis
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.num_classes = num_classes
        self.A_shape = A_shape

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.model = deep_class_model(in_shape=in_shape, out_shape=num_classes)
        elif self.model_type == "linear_model":
            self.model = linear_model(in_shape=in_shape, out_shape=num_classes)
        else:
            raise

        self.pred_loss = cost_pred
        self.clf_optimizer = optim.Adam(self.model.parameters(),lr=self.lr_loss)

        self.lambdas = torch.Tensor([lambda_vec])

        self.dis = class_discriminator(inp=num_classes+self.A_shape+1)
        self.dis_loss = nn.BCELoss(reduce=False)
        self.adv_optimizer = optim.Adam(self.dis.parameters(),lr=self.lr_dis)

        self.epochs = epochs
        self.loss_steps = loss_steps
        self.dis_steps = dis_steps

        self.scaler_x = StandardScaler()
        self.scaler_z = StandardScaler()
        self.scaler_zt = StandardScaler()
        self.scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),
                                                        columns=df.columns,
                                                        index=df.index)


    def fit(self, X, Y, epochs_list = []):
        X_train = pd.DataFrame(data=X[:,self.A_shape:])
        y_train = pd.DataFrame(data=Y)
        orig_Z = X[:,0:self.A_shape]
        Z_train = pd.DataFrame(data=orig_Z)

        log_lik_mat = utility_functions.Class_density_estimation(Y, orig_Z, Y, orig_Z)

        self.scaler_x.fit(X_train)
        X_train = X_train.pipe(self.scale_df, self.scaler_x)
        self.scaler_z.fit(Z_train)
        Z_train = Z_train.pipe(self.scale_df, self.scaler_z)

        self.checkpoint_list = []
        self.cp_model_list = []
        self.cp_dis_list = []

        y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,self.epochs,log_lik_mat))
        Z_perm_index = np.argsort(y_perm_index)
        Z_tilde_list = orig_Z[Z_perm_index]
        for epoch in range(1, self.epochs + 1):
            # print(epoch, ": start")
            Z_tilde = Z_tilde_list[epoch - 1]
            Zt_train = pd.DataFrame(data=Z_tilde)
            self.scaler_zt.fit(Zt_train)
            Zt_train = Zt_train.pipe(self.scale_df, self.scaler_zt)
            train_data = PandasDataSet(X_train, y_train, Z_train, Zt_train)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

            self.model, self.dis = train_classifier(self.model,
                                                    self.dis,
                                                    train_loader,
                                                    self.pred_loss,
                                                    self.dis_loss,
                                                    self.clf_optimizer,
                                                    self.adv_optimizer,
                                                    self.lambdas,
                                                    self.dis_steps,
                                                    self.loss_steps,
                                                    self.num_classes)
           
            if epoch in epochs_list:
                self.checkpoint_list.append(epoch)
                cp_model = copy.deepcopy(self.model)
                cp_dis = copy.deepcopy(self.dis)
                self.cp_model_list.append(cp_model)
                self.cp_dis_list.append(cp_dis)

    def predict(self,X):
        X = X[:,self.A_shape:]
        X_test = pd.DataFrame(data=X)
        X_test = X_test.pipe(self.scale_df, self.scaler_x)

        test_data = PandasDataSet(X_test)

        with torch.no_grad():
            Yhat = self.model(test_data.tensors[0])

        # sm = nn.Softmax(dim=1)
        sm = nn.Softmax(dim=1)
        Yhat = sm(Yhat)
        Yhat = Yhat.detach().numpy()

        return Yhat
##############################################################
##################### Regression Part ########################
##############################################################

def inner_train_adversary_regression(model, dis, x, y, a, at, pred_loss, dis_loss, clf_optimizer, adv_optimizer, lambdas, dis_steps, loss_steps):
    yhat = model(x).detach()
    dis.zero_grad()
    if len(yhat.size())==1:
        yhat = yhat.unsqueeze(dim=1)
    real = torch.cat((yhat,at,y),1)
    fake = torch.cat((yhat,a,y),1)
    in_dis = torch.cat((real, fake), 0)
    out_dis = dis(in_dis)
    labels = torch.cat((torch.ones(real.shape[0],1), torch.zeros(fake.shape[0],1)), 0)
    loss_adv = (dis_loss(out_dis, labels) * lambdas).mean()
    loss_adv.backward()
    adv_optimizer.step()
    return dis

def inner_train_model_regression(model, dis, x, y, a, at, pred_loss, dis_loss, clf_optimizer, adv_optimizer, lambdas, dis_steps, loss_steps):
    yhat = model(x)
    if len(yhat.size())==1:
        yhat = yhat.unsqueeze(dim=1)
    fake = torch.cat((yhat,a,y),1)
    real = torch.cat((yhat,at,y),1)
    in_dis = torch.cat((real, fake), 0)

    model.zero_grad()
    clf_loss = (1.0-lambdas)*pred_loss(yhat.squeeze(), y.squeeze())            
    labels = torch.cat((torch.zeros(real.shape[0],1), torch.ones(fake.shape[0],1)), 0)
    clf_loss += (dis_loss(dis(in_dis), labels) * lambdas).mean()
    clf_loss.backward()
    clf_optimizer.step()
    return model

def train_regressor(model, dis, data_loader, pred_loss, dis_loss,
                    clf_optimizer, adv_optimizer, lambdas, dis_steps, loss_steps):
    # Train adversary
    for i in range(dis_steps):
        for x, y, a, at in data_loader:
            dis = inner_train_adversary_regression(model, dis, x, y, a, at,
                                                   pred_loss, dis_loss,
                                                   clf_optimizer, adv_optimizer,
                                                   lambdas, dis_steps, loss_steps)

    # Train predictor
    for i in range(loss_steps):
        for x, y, a, at in data_loader:
            model = inner_train_model_regression(model, dis, x, y, a, at, pred_loss,
                                                 dis_loss, clf_optimizer,
                                                 adv_optimizer, lambdas,
                                                 dis_steps, loss_steps)

    return model, dis
    
class EquiRegLearner:

    def __init__(self,
                 lr_loss,
                 lr_dis,
                 epochs,
                 loss_steps,
                 dis_steps,
                 cost_pred,
                 in_shape,
                 batch_size,
                 model_type,
                 lambda_vec,
                 out_shape,
                 A_shape,
                 use_standardscaler = True):

        self.lr_loss = lr_loss
        self.lr_dis = lr_dis
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.use_standardscaler = use_standardscaler
        self.A_shape = A_shape

        self.model_type = model_type
        if self.model_type == "deep_model":
            self.model = deep_reg_model(in_shape=in_shape, out_shape=out_shape)
        elif self.model_type == "linear_model":
            self.model = linear_model(in_shape=in_shape, out_shape=out_shape)
        else:
            raise

        self.pred_loss = cost_pred
        self.clf_optimizer = optim.Adam(self.model.parameters(),lr=self.lr_loss) 

        self.lambdas = torch.Tensor([lambda_vec])

        self.dis = reg_discriminator(out_shape + 1 + A_shape)
        self.dis_loss = nn.BCELoss(reduce=False)
        self.adv_optimizer = optim.Adam(self.dis.parameters(),lr=self.lr_dis) 

        self.epochs = epochs
        self.loss_steps = loss_steps
        self.dis_steps = dis_steps

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_z = StandardScaler()
        self.scaler_zt = StandardScaler()

        self.scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),
                                                        columns=df.columns,
                                                        index=df.index)
        


    def fit(self, X, Y, epochs_list = []):
        X_train = pd.DataFrame(data=X[:,self.A_shape:])
        y_train = pd.DataFrame(data=Y)
        orig_Z = X[:,0:self.A_shape]
        Z_train = pd.DataFrame(data=orig_Z)

        log_lik_mat = utility_functions.MAF_density_estimation(Y, orig_Z, Y, orig_Z)

        if self.use_standardscaler:
            self.scaler_x.fit(X_train)
            X_train = X_train.pipe(self.scale_df, self.scaler_x)

            self.scaler_z.fit(Z_train)
            Z_train = Z_train.pipe(self.scale_df, self.scaler_z)

            if self.out_shape==1:
                self.scaler_y.fit(y_train)
                y_train = y_train.pipe(self.scale_df, self.scaler_y)

        self.checkpoint_list = []
        self.cp_model_list = []
        self.cp_dis_list = []

        y_perm_index = np.squeeze(utility_functions.generate_X_CPT(50,self.epochs,log_lik_mat))
        Z_perm_index = np.argsort(y_perm_index)
        Z_tilde_list = orig_Z[Z_perm_index]
        for epoch in range(1, self.epochs + 1):
            # print(epoch, ": start")
            Z_tilde = Z_tilde_list[epoch - 1]
            Zt_train = pd.DataFrame(data=Z_tilde)

            if self.use_standardscaler:
                self.scaler_zt.fit(Zt_train)
                Zt_train = Zt_train.pipe(self.scale_df, self.scaler_zt)

            train_data = PandasDataSet(X_train, y_train, Z_train, Zt_train)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)

            self.model, self.dis = train_regressor(self.model,
                                                     self.dis,
                                                     train_loader,
                                                     self.pred_loss,
                                                     self.dis_loss,
                                                     self.clf_optimizer,
                                                     self.adv_optimizer,
                                                     self.lambdas,
                                                     self.dis_steps,
                                                     self.loss_steps)
            if epoch in epochs_list:
                self.checkpoint_list.append(epoch)
                cp_model = copy.deepcopy(self.model)
                cp_dis = copy.deepcopy(self.dis)
                self.cp_model_list.append(cp_model)
                self.cp_dis_list.append(cp_dis)

    def predict(self,X):
        X = X[:,self.A_shape:]
        X_test = pd.DataFrame(data=X)

        if self.use_standardscaler:
            X_test = X_test.pipe(self.scale_df, self.scaler_x)

        test_data = PandasDataSet(X_test)

        with torch.no_grad():
            Yhat = self.model(test_data.tensors[0]).squeeze().detach().numpy()

        if self.out_shape==1 and self.use_standardscaler:
            out = self.scaler_y.inverse_transform(Yhat.reshape(-1, 1)).squeeze()
        elif self.out_shape==1:
            out = Yhat.squeeze()
        else:
            out = 0*Yhat
            out[:,0] = np.min(Yhat,axis=1)
            out[:,1] = np.max(Yhat,axis=1)

        return out
