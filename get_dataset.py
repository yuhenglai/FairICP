import os
import urllib
import numpy as np
import pandas as pd
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

def read_crimes_data_df(base_path, dim = 1):
    threshold_a = 0.1
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

def get_adult(base_path, dim = 2):
    #This function is a minor modification from https://github.com/jmikko/fair_ERM
    # def load_adult(nTrain=None, scaler=True, shuffle=False):
    nTrain = None
    scaler = True
    shuffle = False
    if shuffle:
        print('Warning: I wont shuffle because adult has fixed test set')
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    # if not os.path.isfile('adult.data'):
    #     urllib.request.urlretrieve(
    #         "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")
    #     urllib.request.urlretrieve(
    #         "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "adult.test")
    data = pd.read_csv(
        base_path + "adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        base_path + "adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
        skiprows=1, header=None
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                    'Married-civ-spouse', 'Married-spouse-absent',
                    'Never-married', 'Separated', 'Widowed'],
                    ['not married', 'married', 'married', 'married',
                    'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    #Care there is a final dot in the class only in test set which creates 4 different classes
    target = np.array([-1.0 if (val == 0 or val==1) else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if nTrain is None:
        nTrain = len_train
    data = namedtuple('_', 'data, target')(datamat, target)

    encoded_data = pd.DataFrame(data.data)
    encoded_data['Target'] = (data.target+1)/2
    gender = 1. * (data.data[:,9]!=data.data[:,9][0])
    race = (data.data[:,8] == data.data[:,8][0]).astype(float)
    age = data.data[:,0]

    if dim == 1:
        df = encoded_data
        X = encoded_data.drop(columns = [9,'Target']).values
        A = gender[:,None]
        Y = encoded_data['Target'].values
    elif dim == 2:
        df = encoded_data
        X = encoded_data.drop(columns = [8, 9,'Target']).values
        A = np.concatenate([gender[:,None], race[:,None]], axis = 1)
        Y = encoded_data['Target'].values

    return df, X, A, Y

def get_ACSIncome(base_path, dim=2):
    # Combine features and labels into one DataFrame
    ca_features = pd.read_csv(os.path.join(base_path, "ca_features.csv"), index_col = [0])
    data = ca_features.copy()
    ca_labels = pd.read_csv(os.path.join(base_path, "ca_labels.csv"), index_col = [0])
    data['income'] = ca_labels
    data['SEX'] = (data['SEX'] == 2).astype(float)
    data['RAC1P'] = (data['RAC1P'] == 2).astype(float)

    # Drop rows with missing values
    data = data.dropna()

    # List of categorical columns
    categorical_cols = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']

    # Encode categorical columns
    for col in categorical_cols:
        unique_values, encoded_values = np.unique(data[col], return_inverse=True)
        data[col] = encoded_values

    # Separate features and target
    features = data.drop('income', axis=1)
    target = data['income'].astype(float).values

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Prepare encoded data DataFrame
    encoded_data = pd.DataFrame(features_scaled, columns=features.columns)
    encoded_data['Target'] = target

    # Prepare sensitive attributes
    # For gender: 1 if Female (SEX == 2), 0 if Male (SEX == 1)
    gender = data['SEX']

    # For race: 1 if Black or African American alone (RAC1P == 2), 0 otherwise
    race = data['RAC1P']

    # For age (already numerical, scaled)
    age = features_scaled[:, features.columns.get_loc('AGEP')]

    # Prepare X, A, Y based on the dimension
    if dim == 1:
        df = encoded_data
        X = encoded_data.drop(columns=['SEX', 'Target']).values
        A = gender[:, None]
        Y = encoded_data['Target'].values
    elif dim == 2:
        df = encoded_data
        X = encoded_data.drop(columns=['SEX', 'RAC1P', 'Target']).values
        A = np.column_stack((gender, race))
        Y = encoded_data['Target'].values
    elif dim == 3:
        df = encoded_data
        X = encoded_data.drop(columns=['AGEP', 'SEX', 'RAC1P', 'Target']).values
        A = np.column_stack((gender, race, age))
        Y = encoded_data['Target'].values
    else:
        df = encoded_data
        X = encoded_data.drop(columns=['Target']).values
        A = None
        Y = encoded_data['Target'].values

    return df, X, A, Y

            
def get_train_test_data(base_path, dataset, seed, dim = 1):
        
    if dataset == "crimes":
        df, X_, A_, Y_ = read_crimes_data_df(base_path, dim = dim)
        n_train = int(Y_.shape[0]*0.6)
        n_train = n_train - n_train%2        

    elif dataset == "compas":
        df, X_, A_, Y_ = read_compas_data_df(base_path, dim = dim)
        n_train = int(Y_.shape[0]*0.6)
        n_train = n_train - n_train%2

    elif dataset == 'adult':
        df, X_, A_, Y_ = get_adult(base_path, dim = dim)
        n_train = int(Y_.shape[0]*0.6)
        n_train = n_train - n_train%2

    elif dataset == 'acs_income':
        df, X_, A_, Y_ = get_ACSIncome(base_path, dim = dim)
        n_train = int(Y_.shape[0]*0.6)
        n_train = n_train - n_train%2

    t0 = np.random.get_state()
    np.random.seed(seed)
    all_inds = np.random.permutation(Y_.shape[0])
    np.random.set_state(t0)

    inds_train = all_inds[:n_train]
    inds_test = all_inds[n_train:]

    X = X_[inds_train]
    A = A_[inds_train]
    Y = Y_[inds_train]
    
    X_test = X_[inds_test]
    A_test = A_[inds_test]
    Y_test = Y_[inds_test]

    return X, A, Y, X_test, A_test, Y_test

def get_full_dataset(base_path, dataset, dim=1):
    if dataset == "crimes":
        df, X_, A_, Y_ = read_crimes_data_df(base_path, dim=dim)
    elif dataset == "compas":
        df, X_, A_, Y_ = read_compas_data_df(base_path, dim=dim)
    elif dataset == 'adult':
        df, X_, A_, Y_ = get_adult(base_path, dim=dim)
    elif dataset == 'acs_income':
        df, X_, A_, Y_ = get_ACSIncome(base_path, dim=dim)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return X_, A_, Y_


