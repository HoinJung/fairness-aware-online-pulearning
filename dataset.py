
import numpy as np
import scipy.sparse as sparse
from torch.utils.data import Dataset
import torch



        
class CustomDataset(Dataset):
    def __init__(self, data, target, indicator, sensitive_attribute):
        self.data = data
        self.target = target
        self.indicator = indicator.astype(int)
        self.sensitive_attribute = sensitive_attribute.astype(int)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.target[index], dtype=torch.float32)
        s = torch.tensor(self.indicator[index], dtype=torch.float32)
        a = torch.tensor(self.sensitive_attribute[index], dtype=torch.float32)
        return x, y, s, a
    

def preprocess_data(dataset,i): 
    f = np.load(f'dataset/{dataset}_{i}_data.npz')
    X_train = f['X_train']
    X_test = f['X_test']
    Y_train = f['Y_train']
    A_train = f['A_train']
    Y_test = f['Y_test']
    A_test = f['A_test']
    if sparse.issparse(X_train):
        X_train = X_train.toarray()
    if sparse.issparse(X_test):
        X_test = X_test.toarray()
    return X_train,X_test, Y_train, Y_test, A_train, A_test