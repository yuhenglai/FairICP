import os
import urllib
import numpy as np
import pandas as pd

def read_crimes_data_df(base_path, dim = 1):
    label='ViolentCrimesPerPop'
    sensitive_attribute='racepctblack'

    if not os.path.isfile(base_path + "communities.data"):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", base_path + "communities.data")
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            base_path + "communities.names")

    # create names
    names = []
    with open(base_path + 'communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv(base_path + 'communities.data', names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    data.drop(to_drop, axis=1, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    to_drop = []
    y = data[label].values
    to_drop += [label]
    y_df = pd.DataFrame({label: y})
    
    z = data[sensitive_attribute].values
    z_df = pd.DataFrame({sensitive_attribute: z})
    to_drop += [sensitive_attribute]

    if dim == 3:
        z = z[:, None]
        multi_list = ["racePctAsian", "racePctHisp"]
        for sensitive_attr_multi in multi_list:
            z = np.concatenate((z,data[sensitive_attr_multi].values[:,None]),1)
            z_df[sensitive_attr_multi] = data[sensitive_attr_multi]
            to_drop += [sensitive_attr_multi]

    data.drop(to_drop, axis=1, inplace=True)
        
    x = np.array(data.values)
    if len(z.shape) == 1: z = z[:, None]
    return pd.concat([y_df, z_df, data], axis = 1), x, z, y

def read_compas_data_df(base_path, dim = 1):
    
    df = pd.read_csv(base_path + 'compas.csv')
    column_names = df.columns
    response_name = "two_year_recid"
    column_names = column_names[column_names!=response_name]
    column_names = column_names[column_names!="Unnamed: 0"]
    df = df[[response_name] + list(column_names)]

    df["age"] = np.log(df["age"] - df["age"].min() + 1)
    df["age"] = (df["age"] - df["age"].mean()) / df["age"].std()
    if dim == 1:
        Y = df[response_name].values
        A = df['race'].values
        X = df.drop(["two_year_recid", "race"], axis=1).values
        if len(A.shape) == 1: A = A[:, None]
        return df, X, A, Y
    elif dim == 2:
        Y = df[response_name].values
        A = df[['race', "sex"]].values
        X = df.drop(["two_year_recid", "race", "sex"], axis=1).values
        return df, X, A, Y

            
def get_train_test_data(base_path, dataset, seed, dim = 1):
        
    if dataset == "crimes":
        df, X_, A_, Y_ = read_crimes_data_df(base_path, dim = dim)
        n_train = int(Y_.shape[0]*0.6)
        n_train = n_train - n_train%2
        n_cal = int( (Y_.shape[0]-n_train) / 2)           

    elif dataset == "compas":
        df, X_, A_, Y_ = read_compas_data_df(base_path, dim = dim)
        n_train = int(Y_.shape[0]*0.6)
        n_train = n_train - n_train%2
        n_cal = int( (Y_.shape[0]-n_train) / 2)    

    t0 = np.random.get_state()
    np.random.seed(seed)
    all_inds = np.random.permutation(Y_.shape[0])
    np.random.set_state(t0)
    
    inds_train = all_inds[:n_train]
    inds_cal = all_inds[n_train:n_train+n_cal]
    inds_test = all_inds[n_train+n_cal:]

    X = X_[inds_train]
    A = A_[inds_train]
    Y = Y_[inds_train]
    
    X_cal = X_[inds_cal]
    A_cal = A_[inds_cal]
    Y_cal = Y_[inds_cal]
    
    X_test = X_[inds_test]
    A_test = A_[inds_test]
    Y_test = Y_[inds_test]

    return X, A, Y, X_cal, A_cal, Y_cal, X_test, A_test, Y_test

