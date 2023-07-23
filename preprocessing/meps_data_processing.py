import numpy as np
import pandas as pd
import os
import sklearn.preprocessing as sk
import argparse
import copy


if __name__ == '__main__':

        
    default_mappings = {
        'label_maps': [{1.0: '>= 10 Visits', 0.0: '< 10 Visits'}],
        'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-White'}]
    }

    def default_preprocessing(df):
        """
        1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
        and 'Non-White' otherwise
        2. Restrict to Panel 21
        3. RENAME all columns that are PANEL/ROUND SPECIFIC
        4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
        5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
        """
        def race(row):
            if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
                return 'White'
            return 'Non-White'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df = df.rename(columns = {'RACEV2X' : 'RACE'})

        df = df[df['PANEL'] == 21]

        # RENAME COLUMNS
        df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                                'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                                'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                                'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                                'POVCAT16' : 'POVCAT', 'INSCOV16' : 'INSCOV'})

        df = df[df['REGION'] >= 0] # remove values -1
        df = df[df['AGE'] >= 0] # remove values -1

        df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

        df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

        def utilization(row):
            return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']

        df['TOTEXP16'] = df.apply(lambda row: utilization(row), axis=1)
        lessE = df['TOTEXP16'] < 10.0
        df.loc[lessE,'TOTEXP16'] = 0.0
        moreE = df['TOTEXP16'] >= 10.0
        df.loc[moreE,'TOTEXP16'] = 1.0

        df = df.rename(columns = {'TOTEXP16' : 'UTILIZATION'})
        return df
    path = 'preprocessing/meps/h192.csv'
    data = pd.read_csv(path, sep=',')
    data = default_preprocessing(data)


    features_to_keep=['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PCS42',
                                 'MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION']
    data = data[features_to_keep]
    data['RACE'] = data['RACE'].map({'White':1, 'Non-White':0})
    dataset = 'meps'
    for i in range(20):
        shuffeld=copy.deepcopy(data)
        shuffled = data.sample(frac=1,random_state=2021).reset_index(drop=True)
        group_label = shuffled['RACE'].to_numpy()
        X_unordered = shuffled.iloc[:, :-1].values
        sensitive_feature = X_unordered[:,3] # (race)
        X = np.hstack((sensitive_feature[..., np.newaxis], X_unordered[:,:3], X_unordered[:,4:]))
        Y = shuffled.iloc[:, -1].values

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