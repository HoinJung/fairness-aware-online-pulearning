import numpy as np
import pandas as pd
import os
import sklearn.preprocessing as sk
import argparse
import copy


if __name__ == '__main__':

        
    print("===============================================")
    print("Preprocessing the Bank Dataset...\n")
    dataset = 'bank'
    # Load .csv file
    path = 'preprocessing/bank/bank-additional-full.csv'

    data = pd.read_csv(path,sep=';')
    # Choose the features as prediction features. We chose the ones provided in the paper (Table 1):
    # ["ID", "age", "gender", "education", "country", "ethnicity", "nscore", "escore", "oscore", "ascore", "cscore", "impulsive", "ss", "coke"]
    data=data[data['marital']!='unknown']
    data['marital'] = data['marital'].map({'married':1,'single':0,'divorced':0})
    data['y'] = data['y'].map({'yes':1,'no':0})
    lessE = data['age'] < 25
    data.loc[lessE,'age'] = 0.0
    moreE = data['age'] >= 25
    data.loc[moreE,'age'] = 1.0

    label_encoder = sk.LabelEncoder()
    attribute_to_label = ['default','housing','loan','contact','month','day_of_week',\
        'poutcome']
    data[attribute_to_label] = data[attribute_to_label].apply(label_encoder.fit_transform)

    attribute_to_onehot = ['job', 'education']
    data = pd.get_dummies(data, columns = attribute_to_onehot)

    for i in range(20):
        shuffeld=copy.deepcopy(data)
        shuffeld = shuffeld.sample(frac=1,random_state=i).reset_index(drop=True)
        
        group_label = shuffeld['age'].to_numpy()
        print(f'group_label shape: {group_label.shape}\n')
        label = shuffeld.pop('y')
        shuffeld = pd.concat([shuffeld,label],1)  
        sensitive_feature = shuffeld.pop('age')
        X_unordered = shuffeld.iloc[:, :-1].values
        X=np.hstack((sensitive_feature[...,np.newaxis],X_unordered))
        Y=shuffeld.iloc[:,-1].values
        
        # Standardize suffled column-wise
        scaler = sk.StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f'Shape of the datapoints:           {X_scaled.shape}')
        print(f'Shape of the corresponding labels: {Y.shape}\n')

        
        idx = round(0.80*len(X_scaled))
        

        ## online validationf
        X_train = X_scaled[:idx]
        X_test = X_scaled[idx:]
        Y_train = Y[:idx]
        Y_test = Y[idx:]
        A_train = group_label[:idx]
        A_test = group_label[idx:]
        print(f'X_train shape: {X_train.shape}')
        print(f'X_test shape:  {X_test.shape}')
        print(f'Y_train shape: {Y_train.shape}')
        print(f'Y_test shape:  {Y_test.shape}')
        print(f'A_train shape: {A_train.shape}')
        print(f'A_test shape:  {A_test.shape}')

        # Create output folder if it doesn't exist
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        np.savez_compressed(f'dataset/{dataset}_{i}_data.npz', X_train=X_train,  X_test=X_test, Y_train=Y_train, Y_test=Y_test, A_train=A_train, A_test = A_test)
        print("===============================================")