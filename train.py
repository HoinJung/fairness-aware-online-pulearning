import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset,preprocess_data
from model import MLP, Linear
from utils import evaluate,risk_estimator,lower_upper_loss,make_pu
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split


def train(args,model,train_loader,optimizer):
    args.beta_pu = 0
    args.gamma_pu = 1
    train_loss = 0
    model.train()
    device = args.device
    for batch_idx, (data, target,indicator,sensitive) in enumerate(train_loader):
        data, target, indicator = data.to(device), target.to(device), indicator.to(device)
        optimizer.zero_grad()
        sensitive[sensitive==0]=-1
        output = model(data)
        output = torch.squeeze(output, 1)
        loss = risk_estimator(args,args.loss_func, output,indicator)
        if args.fair:
            fair_loss = lower_upper_loss(args, target, output, sensitive)
            loss+=fair_loss
        train_loss += loss.item()   
        loss.backward()
        optimizer.step()
        
def test(epoch, model, X_test,Y_test,A_test):
    model.eval()
    with torch.no_grad():
        output_test = model(torch.tensor(X_test).cuda().float())
        if epoch%1==0:            
            Acc_test, dp_gap_test, eo_gap_test = evaluate(output_test, Y_test, A_test)
    return Acc_test,dp_gap_test, eo_gap_test
    


def run_train(args):
    dataset = args.dataset
    model_name = args.model
    r1 = args.r1
    batch_size = args.batch_size
    lr = args.lr
    opt = args.opt
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = []
    
    test_dp = []
    test_eod = []
    
    for i in tqdm(range(args.num_exp)):
        X_train,  X_test, Y_train,  Y_test, A_train,A_test = preprocess_data(dataset,i)
        s_train = make_pu(X_train, Y_train, r1)
        s_test = make_pu(X_test, Y_test, r1)
        prior = torch.tensor(len(Y_train[Y_train==1])/len(Y_train))
        args.prior = prior
        Y_train[Y_train==0]=-1
        s_train[s_train==0]=-1
        Y_test[Y_test==0]=-1
        s_test[s_test==0]=-1
        
        
        X_train,X_val, Y_train,Y_val,s_train,s_val,A_train,A_val =  train_test_split(X_train, Y_train,s_train,A_train, test_size=0.2, random_state=42)
        
        train_dataset = CustomDataset(X_train, Y_train,s_train,A_train)        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        N = len(Y_train)   
        Y_train = torch.tensor(Y_train).cuda().float()
        sensitive = torch.tensor(A_train).cuda().float()
        n_priv = torch.sum(sensitive==1)
        p1 = n_priv / N
        p0 = 1 - p1
        sensitive[sensitive==0]=-1
        n_priv_pos = Y_train[torch.where(sensitive==1)[0]]==1
        n_priv_pos_sum = torch.sum(n_priv_pos)
        p11 = n_priv_pos_sum / N
        p01 = prior-p11
        n_priv_neg = Y_train[torch.where(sensitive==1)[0]]==-1
        n_priv_neg_sum = torch.sum(n_priv_neg)
        p10 = n_priv_neg_sum / N
        p00 = 1-prior-p10
        args.probs = (p0,p1,p11,p01,p10,p00)
        input_size = len(X_train[1])
        if args.loss_type == 'log':
            args.loss_func=(lambda x: torch.sigmoid(-x))
        elif args.loss_type == 'dh':
            args.loss_func=lambda x: torch.max( -x, torch.max(torch.tensor(0.), (1-x)/2))
        elif args.loss_type == 'sq':
            args.loss_func=(lambda x: 1/4*(x-1)**2  - 1/4)
        
        
        if model_name=='mlp':
            model = MLP(input_size=input_size).cuda()    
        elif model_name=='linear':
            model = Linear(input_size=input_size).cuda()
        
        if opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-1)
        elif opt =='adam':
            optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
        
        best_acc = 0  
        best_result = [0,0,0]
        for epoch in range(args.epoch):
            
            train(args,model,train_loader,optimizer)
            curr_test_result = test(epoch, model, X_test,Y_test,A_test)
            curr_val_result = test(epoch, model, X_val,Y_val,A_val)
            acc_val = curr_val_result[0]
            if args.verbose:
                print(f"ACC: {curr_val_result[0]:.4f}, DP: {curr_val_result[1]:.4f}, EOD: {curr_val_result[2]:.4f}")
            if best_acc<=acc_val:
                best_acc = acc_val
                best_result = curr_test_result
        test_acc.append(best_result[0])
        test_dp.append(best_result[1])
        test_eod.append(best_result[2])

    print(f"Test Result. ACC:{np.nanmean(test_acc):.4f}/{np.nanstd(test_acc):.4f}, DP:{np.nanmean(test_dp):.4f}/{np.nanstd(test_dp):.4f}, EOD:{np.nanmean(test_eod):.4f}/{np.nanstd(test_eod):.4f}")
    
    