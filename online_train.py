
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
import torch
from torch.utils.data import DataLoader,Subset
from dataset import CustomDataset,preprocess_data
from model import Linear,O_MLP,O_LSTM
import random
from utils import evaluate,risk_estimator,lower_upper_loss,make_pu,custom_loss, seed_torch
import copy
from tqdm import tqdm
from torch.nn.parameter import Parameter
import random
def pu_risk_OGD(args, t, sensitive, model, data, indicator, params, prev_params,target,baselines):
    output = model(data)
    output = torch.squeeze(output, 1)
    if args.pu_type.lower() in ['pn']:    
        loss = custom_loss(output,target)
    elif args.pu_type.lower() in ['upu', 'nnpu']:
        loss = risk_estimator(args,args.loss_func, output,indicator)
    
    sensitive[sensitive==0]=-1
    
    loss += 0.5 * args.lam * sum(torch.norm(param, p=2)**2 for param in params)
    if t!=0:
        loss += 0.5 * (args.gamma_round + args.lam*(t+1)) * sum(torch.norm(param - prev,p=2)**2 for param, prev in zip(params, prev_params))
    if args.fair:
        fair_loss,baselines = lower_upper_loss(args, target, output, sensitive,baselines)
        loss+=fair_loss
    model.zero_grad()    
    loss.backward()
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            param -= args.eta * param.grad
            param.grad.zero_()
    prev_params = [param.clone().detach() for param in model.parameters()]
    return prev_params,baselines

def pu_risk_lstm(args, t, sensitive, model, data, indicator,target,baselines):
    
    predictions_per_layer  = model(data)
    losses_per_layer = []
    
    for i, output in  enumerate(predictions_per_layer):
        output = torch.squeeze(output, 1)
        if args.pu_type.lower() in ['pn']:    
            loss = custom_loss(output,target)
        elif args.pu_type.lower() in ['upu', 'nnpu']:
            loss = risk_estimator(args,args.loss_func, output,indicator)
        if args.fair:
            fair_loss,baselines = lower_upper_loss(args, target, output, sensitive,baselines)
            loss+=fair_loss
        losses_per_layer.append(loss)
        
        w_hh = [None] * len(losses_per_layer)
        b_hh = [None] * len(losses_per_layer)
        w = [None] * len(losses_per_layer)
        b = [None] * len(losses_per_layer)

        with torch.no_grad():
            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph=True)

                model.output_layers[i].weight.data -= args.eta * model.alpha[i] * model.output_layers[i].weight.grad.data
                model.output_layers[i].bias.data -= args.eta * model.alpha[i] * model.output_layers[i].bias.grad.data

                for j in range(i + 1):
                    if w[j] is None:
                        w[j] = model.alpha[i] * model.lstm_layers[j].weight_ih.grad.data
                        b[j] = model.alpha[i] * model.lstm_layers[j].bias_ih.grad.data
                        w_hh[j] = model.alpha[i] * model.lstm_layers[j].weight_hh.grad.data
                        b_hh[j] = model.alpha[i] * model.lstm_layers[j].bias_hh.grad.data
                    else:
                        w[j] += model.alpha[i] * model.lstm_layers[j].weight_ih.grad.data
                        b[j] += model.alpha[i] * model.lstm_layers[j].bias_ih.grad.data
                        w_hh[j] += model.alpha[i] * model.lstm_layers[j].weight_hh.grad.data
                        b_hh[j] += model.alpha[i] * model.lstm_layers[j].bias_hh.grad.data

            model.zero_gradient()
            
            for i in range(len(losses_per_layer)):
                model.lstm_layers[i].weight_ih.data -= args.eta * w[i]
                model.lstm_layers[i].bias_ih.data -= args.eta * b[i]
                model.lstm_layers[i].weight_hh.data -= args.eta * w_hh[i]
                model.lstm_layers[i].bias_hh.data -= args.eta * b_hh[i]

            for i in range(len(losses_per_layer)):
                model.alpha[i] *= torch.pow(model.b, losses_per_layer[i])
                model.alpha[i] = torch.max(
                    model.alpha[i], model.s / model.max_num_hidden_layers)
            
            z_t = torch.sum(model.alpha)
            model.alpha = Parameter(
                    model.alpha / z_t, requires_grad=False).to(model.device)
            
            params_list = []
            for i in range(len(losses_per_layer)):
                params_list.append([
                    model.lstm_layers[i].weight_ih.data,
                    model.lstm_layers[i].bias_ih.data,
                    model.lstm_layers[i].weight_hh.data,
                    model.lstm_layers[i].bias_hh.data,
                    model.output_layers[i].weight.data,
                    model.output_layers[i].bias.data
                ])

        return params_list,baselines
def pu_risk_ONN(args, t, sensitive, model, data, indicator,target,baselines):
    
    predictions_per_layer  = model(data)
    losses_per_layer = []
    sensitive[sensitive==0]=-1
    
    for i, output in  enumerate(predictions_per_layer):
        output = torch.squeeze(output, 1)
        if args.pu_type.lower() in ['pn']:    
            loss = custom_loss(output,target)
        elif args.pu_type.lower() in ['upu', 'nnpu']:
            loss = risk_estimator(args,args.loss_func, output,indicator)
        if args.fair:
            fair_loss,baselines = lower_upper_loss(args, target, output, sensitive,baselines)
            loss+=fair_loss
        losses_per_layer.append(loss)
    w = [None] * len(losses_per_layer)
    b = [None] * len(losses_per_layer)
    with torch.no_grad():
        for i in range(len(losses_per_layer)):
            losses_per_layer[i].backward(retain_graph=True)
            model.output_layers[i].weight.data -= args.eta * \
                                                model.alpha[i] * model.output_layers[i].weight.grad.data                 
            model.output_layers[i].bias.data -= args.eta * \
                                                model.alpha[i] * model.output_layers[i].bias.grad.data
            for j in range(i + 1):
                  if w[j] is None:
                      w[j] = model.alpha[i] * model.hidden_layers[j].weight.grad.data
                      b[j] = model.alpha[i] * model.hidden_layers[j].bias.grad.data
                  else:
                      w[j] += model.alpha[i] * model.hidden_layers[j].weight.grad.data
                      b[j] += model.alpha[i] * model.hidden_layers[j].bias.grad.data
        model.zero_gradient()    
        for i in range(len(losses_per_layer)):
            model.hidden_layers[i].weight.data -= args.eta * w[i]
            model.hidden_layers[i].bias.data -= args.eta * b[i]
        for i in range(len(losses_per_layer)):
            model.alpha[i] *= torch.pow(model.b, losses_per_layer[i])
            model.alpha[i] = torch.max(
                model.alpha[i], model.s / model.max_num_hidden_layers)
    z_t = torch.sum(model.alpha)
    model.alpha = Parameter(
            model.alpha / z_t, requires_grad=False).to(model.device)
    params_list = []
    for i in range(len(losses_per_layer)):
        params_list.append([model.hidden_layers[i].weight.data ,\
                            model.hidden_layers[i].bias.data,model.output_layers[i].weight.data,model.output_layers[i].bias.data ])
    return params_list,baselines

def train(args,t,model,online_loader,prev_params,baselines):
    args.beta_pu = 0
    args.gamma_pu = 1
    args.gamma_round = math.log(args.round)/args.N
    
    args.lam = args.b/args.lr
    args.eta = args.b/(args.lam*((t+1)**(1/2)))
    device = args.device
    
    params = copy.deepcopy(list(model.parameters()))
    for batch_idx, (data, target,indicator,sensitive) in enumerate(online_loader):
        data, target, indicator= data.to(device), target.to(device), indicator.to(device)
        model.train()
        if args.model == 'linear':
            prev_params , baselines= pu_risk_OGD(args, t, sensitive, model, data, indicator, params, prev_params,target,baselines)
        elif args.model=='mlp':
            prev_params,baselines = pu_risk_ONN(args, t, sensitive, model, data, indicator, target,baselines)
        elif args.model == 'lstm':
            prev_params, baselines= pu_risk_lstm(args, t, sensitive, model, data, indicator, target,baselines)
        elif args.model in  ['bert','distill']:
            prev_params , baselines= pu_risk_OGD(args, t, sensitive, model, data, indicator, params, prev_params,target,baselines)
    return prev_params,baselines

def test(args,r, model, X,Y,A):
    model.eval()
    with torch.no_grad():    
        if args.model =='linear':
            output_test = model(torch.tensor(X).cuda().float())
        elif args.model in  ['bert','distill']:
            output_test = model(torch.tensor(X).cuda().float())
        elif args.model in ['mlp','lstm']:
            output_test = torch.sum(torch.mul(
                model.alpha.view(model.max_num_hidden_layers, 1).repeat(1, len(X)).view(
                    model.max_num_hidden_layers, len(X), 1), model(torch.tensor(X).cuda().float())), 0)
    if r%1==0:            
        Acc_test, dp_gap_test, eo_gap_test,f1, confusion, recalls = evaluate(output_test, Y, A)
                
    return Acc_test, dp_gap_test, eo_gap_test,f1, confusion, recalls


def run_online_train(args):
    dataset = args.dataset
    model_name = args.model
    r1 = args.r1
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = []
    test_dp = []
    test_f1 = []
    test_eod = []
    test_conf=[] 
    test_recall = []
    test_recall0 = []
    test_recall1 = []
    for i in tqdm(range(args.num_exp)):
        baselines=None
        seed_torch(i)
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
        if int(args.batch_size) == 1 :    
            args.round = len(X_train)     
        else : 
            batch_size = 999999 
        s_train = make_pu(X_train, Y_train, r1)
        s_test = make_pu(X_test, Y_test, r1)
        prior = torch.tensor(len(Y_train[Y_train==1])/len(Y_train))
        args.prior = prior
        batch_size = args.batch_size 
        
        train_dataset = CustomDataset(X_train, Y_train,s_train,A_train)
        indices = torch.randperm(len(train_dataset))
        N = len(Y_train)
        args.N = N 
        Y_train = torch.tensor(Y_train).cuda().float()
        sensitive = torch.tensor(A_train).cuda().float()

        sensitive[sensitive==0]=-1
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
        args.b = len(train_dataset)//args.round
        subset_indices = [indices[j:j+args.b] for j in range(0, len(train_dataset), args.b)]
        if len(subset_indices)!=args.round:
            subset_indices = subset_indices[:-1] 
            
        random.shuffle(subset_indices)
        subsets = [Subset(train_dataset, indices) for indices in subset_indices]
        
        input_size = len(X_train[1])
        if args.loss_type == 'log':
            args.loss_func=(lambda x: torch.sigmoid(-x))
        elif args.loss_type == 'dh':
            args.loss_func=lambda x: torch.max( -x, torch.max(torch.tensor(0.), (1-x)/2))
        elif args.loss_type == 'sq':
            args.loss_func=(lambda x: 1/4*(x-1)**2  - 1/4)
        if model_name=='mlp':
            node_size = args.hidden_dim
            L = 2
            hidden_nodes = [int(node_size) for i in range(L)]
            model = O_MLP(input_size=input_size, max_num_hidden_layers=len(hidden_nodes),hidden_nodes=hidden_nodes, device=args.device).cuda()    
        elif model_name=='linear':
            model = Linear(input_size=input_size, mode='online').cuda()
        elif model_name == 'lstm':
            node_size = args.hidden_dim
            L = 2
            hidden_nodes = [int(node_size) for i in range(L)]
            model = O_LSTM(input_size=input_size, max_num_hidden_layers=len(hidden_nodes),hidden_nodes=hidden_nodes, device=args.device).cuda()    
        elif model_name in  ['bert','distill']:
            model= Linear(input_size=input_size, mode='online').cuda()
        prev_params = None
        best_f1 = 0
        best_test_result = None
        for t in range(args.round):
            online_loader = DataLoader(subsets[t], batch_size = batch_size, shuffle=True)
            prev_params, baselines = train(args, t, model, online_loader, prev_params, baselines)
            val_result = test(args, t, model, X_val, Y_val, A_val)
            val_f1 = val_result[3]
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_test_result = test(args, t, model, X_test, Y_test, A_test)
        curr_result = best_test_result
        if curr_result is None:
            raise ValueError("The model didn't converge.")
        test_acc.append(curr_result[0])
        test_dp.append(curr_result[1])
        test_eod.append(curr_result[2])
        test_f1.append(curr_result[3])
        test_conf.append(curr_result[4])
        test_recall.append(curr_result[-1][0])
        test_recall0.append(curr_result[-1][1])
        test_recall1.append(curr_result[-1][2])
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



    
    