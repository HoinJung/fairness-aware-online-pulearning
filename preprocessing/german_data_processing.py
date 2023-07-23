import numpy as np
import pandas as pd
import os
import sklearn.preprocessing as sk
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fake', default=False, type=bool)
    parser.add_argument('--fake_num', default=1, type=int)
    args = parser.parse_args()

    """
    Preprocessing of the German Credit Dataset
    http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 
    The script preprocesses the German Credit Dataset and saves the .npz files in Fairness_attack/data
    """

    print("===============================================")
    print("Preprocessing the German Credit Dataset...\n")

    # Load .csv file    
    path = 'preprocessing/german/german_final.csv'
    origin_data = pd.read_csv(path, header=None)
    dataset = 'german'
    for i in range(20):
        data = origin_data.sample(frac=1,random_state=i).reset_index(drop=True)
        # Map categorical/qualitative attributes to numerical ones (One-hot Encoding)
        attributes_to_encode = [0,2,3,5,6,9,11,13,14,16,18,19]
        data = pd.get_dummies(data, columns=attributes_to_encode)

        # Group classes (i.e. [A91, A93, A94] as male (0), [A92, A95] as female (1))
        # data[8] = data[8].map({'A91':0, 'A92':1, 'A93':0, 'A94':0, 'A95':1})
        data[8] = data[8].map({'A91':0, 'A92':1, 'A93':0, 'A94':0, 'A95':1,0:0,1:1})

        # To increase readibility, map a good risk value (1) to 0 and a bad risk value (2) to 1
        data[20] = data[20].map({1: 0, 2:1})


        # Create advantaged and disadvantaged groups: if it's a male (1) map to 0, if it's a female (2) map to 1
        group_label = data[8].to_numpy()
        print(f'group_label shape: {group_label.shape}\n')

        # Move label column to last column
        label = data.pop(20)
        data = pd.concat([data, label], 1)

        # Split to data points and ground truths
        X_unordered = data.iloc[:, :-1].values

        # Move the sensitive feature to index 0 so that it is selected by default
        sensitive_feature = X_unordered[:,3] # (gender)
        X = np.hstack((sensitive_feature[..., np.newaxis], X_unordered[:,:3], X_unordered[:,4:]))
        Y = data.iloc[:, -1].values

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