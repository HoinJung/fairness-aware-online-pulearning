
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math
import torch
from torch.utils.data import DataLoader,Subset
from dataset import CustomDataset,preprocess_data
from model import Linear,O_MLP
import random
from utils import evaluate,risk_estimator,lower_upper_loss,make_pu
import copy
from tqdm import tqdm
from torch.nn.parameter import Parameter


def pu_risk_OGD(args, t, sensitive, model, data, indicator, params, prev_params,target):
    output = model(data)
    output = torch.squeeze(output, 1)
    loss = risk_estimator(args,args.loss_func, output,indicator)
    
    sensitive[sensitive==0]=-1
    
    loss += 0.5 * args.lam * sum(torch.norm(param, p=2)**2 for param in params)
    if t!=0:
        loss += 0.5 * (args.gamma_round + args.lam*(t+1)) * sum(torch.norm(param - prev,p=2)**2 for param, prev in zip(params, prev_params))
    if args.fair:
        fair_loss = lower_upper_loss(args, target, output, sensitive)
        loss+=fair_loss
    model.zero_grad()    
    loss.backward()
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            param -= args.eta * param.grad
            param.grad.zero_()
    prev_params = [param.clone().detach() for param in model.parameters()]
    return prev_params

def pu_risk_ONN(args, t, sensitive, model, data, indicator,target):
    
    predictions_per_layer  = model(data)
    losses_per_layer = []
    sensitive[sensitive==0]=-1
    
    for i, output in  enumerate(predictions_per_layer):
        output = torch.squeeze(output, 1)
        loss = risk_estimator(args,args.loss_func, output,indicator)
        if args.fair:
            fair_loss = lower_upper_loss(args, target, output, sensitive)
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
    return params_list

def train(args,t,model,online_loader,prev_params):
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
            prev_params = pu_risk_OGD(args, t, sensitive, model, data, indicator, params, prev_params,target)
        elif args.model=='mlp':
            prev_params = pu_risk_ONN(args, t, sensitive, model, data, indicator,target)
    return prev_params

def test(args,r, model, X_test,Y_test,A_test):
    model.eval()
    with torch.no_grad():    
        if args.model =='linear':
            output_test = model(torch.tensor(X_test).cuda().float())
        elif args.model =='mlp':
            output_test = torch.sum(torch.mul(
                model.alpha.view(model.max_num_hidden_layers, 1).repeat(1, len(X_test)).view(
                    model.max_num_hidden_layers, len(X_test), 1), model(torch.tensor(X_test).cuda().float())), 0)
    if r%1==0:            
        Acc_test, dp_gap_test, eo_gap_test = evaluate(output_test, Y_test, A_test)
                
    return Acc_test, dp_gap_test, eo_gap_test


def run_online_train(args):
    dataset = args.dataset
    model_name = args.model
    r1 = args.r1
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = []
    test_dp = []
    test_eod = []
    for i in tqdm(range(args.num_exp)):
        
        X_train,  X_test, Y_train,  Y_test, A_train,A_test = preprocess_data(dataset,i)
        if int(args.batch_size) == 1 :    
            args.round = len(X_train)     
        else : 
            batch_size = 999999 
        s_train = make_pu(X_train, Y_train, r1)
        s_test = make_pu(X_test, Y_test, r1)
        prior = torch.tensor(len(Y_train[Y_train==1])/len(Y_train))
        args.prior = prior
        Y_train[Y_train==0]=-1
        Y_test[Y_test==0]=-1
        s_train[s_train==0]=-1
        s_test[s_test==0]=-1  
    
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
            node_size = 128
            L = 2
            hidden_nodes = [int(node_size) for i in range(L)]
            model = O_MLP(input_size=input_size, hidden_feature=64, max_num_hidden_layers=len(hidden_nodes),hidden_nodes=hidden_nodes, device=args.device).cuda()    
        elif model_name=='linear':
            model = Linear(input_size=input_size, mode='online').cuda()
        prev_params = None
        
        for t in range(args.round):
            online_loader = DataLoader(subsets[t], batch_size = batch_size, shuffle=True)
            prev_params = train(args,t, model, online_loader, prev_params)
            curr_result = test(args, t, model, X_test, Y_test, A_test)
            if args.verbose:
                print(f"ACC: {curr_result[0]:.4f}, DP: {curr_result[1]:.4f}, EOD: {curr_result[2]:.4f}")
        test_acc.append(curr_result[0])
        test_dp.append(curr_result[1])
        test_eod.append(curr_result[2])

    print(f"Test Result. ACC:{np.nanmean(test_acc):.4f}/{np.nanstd(test_acc):.4f},  DP:{np.nanmean(test_dp):.4f}/{np.nanstd(test_dp):.4f}, EOD:{np.nanmean(test_eod):.4f}/{np.nanstd(test_eod):.4f}")
    
    

    
    