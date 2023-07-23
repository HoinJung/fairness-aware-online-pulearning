import numpy as np
import pandas as pd
import os
import sklearn.preprocessing as sk
import argparse
import copy
"""
Preprocessing of the Drug Consumption Dataset
http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29#
The script preprocesses the Drug Consumption Dataset and saves the .npz files in Fairness_attack/data
"""

if __name__ == '__main__':

    print("===============================================")
    print("Preprocessing the DRUG Consumption Dataset...\n")
    dataset = 'drug'
    # Load .csv file
    path = 'preprocessing/drug/drug_consumption.data'

    # Choose the features as prediction features. We chose the ones provided in the paper (Table 1):
    # ["ID", "age", "gender", "education", "country", "ethnicity", "nscore", "escore", "oscore", "ascore", "cscore", "impulsive", "ss", "coke"]
    data = pd.read_csv(path, header=None, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,20])



    
    # data[23] = data[23].map({'CL0':1, 'CL1':0, 'CL2':0, 'CL3':0, 'CL4':0, 'CL5':0, 'CL6':0}) #heroin
    data[20] = data[20].map({'CL0':1, 'CL1':0, 'CL2':0, 'CL3':0, 'CL4':0, 'CL5':0, 'CL6':0}) #coke

    # Force int type to labels
    data[20] = data[20].astype(int)
    for i in range(20):
        shuffled=copy.deepcopy(data)
        shuffled = shuffled.sample(frac=1,random_state=i).reset_index(drop=True)

        # Create advantaged and disadvantaged groups
        group_label = shuffled[2].to_numpy()
        group_label = np.where(group_label<0.2,0,1)
        # Map -0.4826 (Male) to 0
        # Map 0.48246 (Female) to 1
        # group_label = np.where(data[5]==-0.31685,1,0)

        # Split to data points and ground truths
        X_unordered = shuffled.iloc[:, :-1].values

        # Move the sensitive feature to index 0 so that it is selected by default
        sensitive_feature = X_unordered[:,2] # (gender)
        X = np.hstack((sensitive_feature[..., np.newaxis], X_unordered[:,:2], X_unordered[:,3:]))
        Y = shuffled.iloc[:, -1].values

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