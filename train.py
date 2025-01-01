import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset,preprocess_data
from model import MLP, Linear, LSTM
from utils import evaluate,risk_estimator,lower_upper_loss,make_pu, custom_loss
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split

def train(args,epoch, model,train_loader,optimizer):
    args.beta_pu = 0
    args.gamma_pu = 1
    train_loss = 0
    model.train()
    device = args.device
    pu = args.pu_type.lower()
    baselines=None
    for batch_idx, (data, target,indicator,sensitive) in enumerate(train_loader):
        data, target, indicator = data.to(device), target.to(device), indicator.to(device)
        output = model(data)
        try :
            output = torch.squeeze(output, 1)
        except : 
            pass
        if pu in ['pn']:
            loss = custom_loss(output,target)
        elif pu in ['upu', 'nnpu']:
            loss = risk_estimator(args,args.loss_func, output,indicator)

        if args.fair:
            fair_loss,baselines = lower_upper_loss(args, target, output, sensitive,baselines)
            loss+=fair_loss
        train_loss += loss.item()   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def test(args, epoch, model, X_test, Y_test, A_test, pu):
    model.eval()
    with torch.no_grad():
        output_test = model(torch.tensor(X_test).cuda().float())
        
        # Determine the loss based on the value of `pu`
        if pu in ['pn']:
            loss = custom_loss(output_test, torch.tensor(Y_test).cuda().float())
        elif pu in ['upu', 'nnpu']:
            loss = risk_estimator(args, args.loss_func, output_test, torch.tensor(X_test).cuda().float())
        else:
            loss = torch.tensor(0)
        if epoch % 1 == 0:
            Acc_test, dp_gap_test, eo_gap_test, f1, confusion, recalls = evaluate(output_test, Y_test, A_test)
    return Acc_test, dp_gap_test, eo_gap_test, f1, confusion, loss.item(), recalls
    


def run_train(args):
    dataset = args.dataset
    model_name = args.model
    r1 = args.r1
    batch_size = args.batch_size
    lr = args.lr
    opt = args.opt
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = []
    test_f1 = []
    test_dp = []
    test_eod = []
    test_conf=[]  
    test_recall = []
    test_recall0 = []
    test_recall1 = []  
    
    for i in tqdm(range(args.num_exp)):
        X_train, X_val,  X_test, Y_train,Y_val,  Y_test, A_train,A_val, A_test = preprocess_data(args,dataset,i)

        if args.model in ['bert','distill']:
            if dataset == 'nela':
                f = np.load(f'{args.dataset}_{args.model}_embeddings.npz')
                X_train = f['X_train']
                X_val = f['X_val']
                X_test = f['X_test']
                f = np.load(f'{args.dataset}_{args.model}_labels.npz')
            else : 
                def load_tokenized_data(file_prefix):
                    data = np.load(f'{file_prefix}_{args.model}_embeddings.npz')
                    return data['embeddings']
                # Load tokenized data for train, val, and test sets
                X_train  = load_tokenized_data(f'{args.dataset}_X_train')
                X_val = load_tokenized_data(f'{args.dataset}_X_val')
                X_test = load_tokenized_data(f'{args.dataset}_X_test')
                
                f = np.load(f'{dataset}_{args.model}_matrix.npz')
            Y_val = f['Y_val']
            A_val = f['A_val']
            A_val = A_val.astype(np.int16)
            Y_train = f['Y_train']
            A_train = f['A_train']
            Y_test = f['Y_test']
            A_test = f['A_test']
            A_test = A_test.astype(np.int16)
            A_val = A_val.astype(np.int16)
            A_train = A_train.astype(np.int16)
            Y_train[Y_train==0]=-1
            Y_test[Y_test==0]=-1
            Y_val[Y_val==0]=-1
            A_train[A_train==0]=-1
            A_val[A_val==0]=-1
            A_test[A_test==0]=-1
        s_train = make_pu(X_train, Y_train, r1)
        s_test = make_pu(X_test, Y_test, r1)
        prior = torch.tensor(len(Y_train[Y_train==1])/len(Y_train))
        args.prior = prior
        train_dataset = CustomDataset(X_train, Y_train,s_train,A_train)        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        N = len(Y_train)   
        Y_train = torch.tensor(Y_train).cuda().float()
        sensitive = torch.tensor(A_train).cuda().float()
        n_priv = torch.sum(sensitive==1)
        p1 = n_priv / N
        p0 = 1 - p1
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
            model = MLP(input_size=input_size, hidden_dim = args.hidden_dim).cuda()    
        elif model_name=='linear':
            model = Linear(input_size=input_size,mode='offline').cuda()
        elif model_name=='lstm':
            model = LSTM(input_size=input_size, hidden_dim = args.hidden_dim).cuda()
        elif model_name in  ['bert','distill']:
            model = Linear(input_size=input_size,mode='offline').cuda()
        if opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-1)
        elif opt =='adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        baselines=None
        best_acc = 0  
        best_f1 = 0
        best_result = None
        for epoch in range(args.epoch):
            
            train(args,epoch,model,train_loader,optimizer)
            curr_test_result = test(args,epoch, model, X_test,Y_test,A_test,args.pu_type)
            curr_val_result = test(args,epoch, model, X_val,Y_val,A_val,args.pu_type)
            f1_val = curr_val_result[3]
            if args.verbose:
                print(f"ACC: {curr_val_result[0]:.4f}, F1: {curr_val_result[3]:.4f}, Recall: {curr_val_result[-1][0]}, DP: {curr_val_result[1]:.4f}, EOD: {curr_val_result[2]:.4f}")
            if best_f1<=f1_val:
                best_epoch = epoch
                best_f1 =f1_val
                best_result = curr_test_result
        test_acc.append(best_result[0])
        test_dp.append(best_result[1])
        test_eod.append(best_result[2])
        test_f1.append(best_result[3])
        test_conf.append(best_result[4])
        test_recall.append(best_result[-1][0])
        test_recall0.append(best_result[-1][1])
        test_recall1.append(best_result[-1][2])

    test_acc_mean = round(np.nanmean(test_acc),4)
    test_acc_std = round(np.nanstd(test_acc),4)
    test_dp_mean = round(np.nanmean(test_dp),4)
    test_dp_std = round(np.nanstd(test_dp),4)
    test_eod_mean = round(np.nanmean(test_eod),4)
    test_eod_std = round(np.nanstd(test_eod),4)
    test_f1_mean = round(np.nanmean(test_f1),4)
    test_f1_std = round(np.nanstd(test_f1),4)

    test_recall_mean = round(np.nanmean(test_recall),4)
    test_recall_std = round(np.nanstd(test_recall),4)
    test_recall0_mean = round(np.nanmean(test_recall0),4)
    test_recall0_std = round(np.nanstd(test_recall0),4)
    test_recall1_mean = round(np.nanmean(test_recall1),4)
    test_recall1_std = round(np.nanstd(test_recall1),4)

    print(f"Test Result. ACC:{test_acc_mean}/{test_acc_std}, F1:{test_f1_mean}/{test_f1_std}, DP:{test_dp_mean}/{test_dp_std}, EOd:{test_eod_mean}/{test_eod_std}")
    print(f"Test Result. Recall:{test_recall_mean}/{test_recall_std}, Recall0:{test_recall0_mean}/{test_recall0_std}, Recall1:{test_recall1_mean}/{test_recall1_std}")
    
    