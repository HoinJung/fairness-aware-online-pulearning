import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score
import torch
import copy
import random
import os
from torch.nn import functional as F
def custom_loss(y_true, y_pred):
    x = y_true * y_pred
    loss = torch.max(-x, torch.max(torch.tensor(0.), (1-x)/2))
    return loss.mean()

def performance_metrics(true_labels, predicted_labels):
    
    TP = np.sum((true_labels == 1) & (predicted_labels == 1))
    FP = np.sum((true_labels == -1) & (predicted_labels == 1))
    TN = np.sum((true_labels == -1) & (predicted_labels == -1))
    FN = np.sum((true_labels == 1) & (predicted_labels == -1))

    # return round(TPR,4), round(FPR,4), round(TNR,4), round(FNR,4)
    return round(TP,4), round(FP,4), round(TN,4), round(FN,4)

def evaluate(output, y_test, A_test):

    y = copy.deepcopy(y_test)
    
    try : 
        pred = torch.squeeze(torch.where(output>=0,1,-1),1).data.cpu().numpy()
    except : 
        pred = np.where(output>=0,1,-1)
    
    
    idx_0 = np.where(A_test==-1)[0]
    idx_1 = np.where(A_test==1)[0]
    # print(pred.shape)
    pred_1 = pred[idx_1]==1
    pred_0 = pred[idx_0]==1
    # print(y_test,pred)
    
    TP_0, FP_0, TN_0, FN_0 = performance_metrics(y_test[idx_0],pred[idx_0])
    TP_1, FP_1, TN_1, FN_1 = performance_metrics(y_test[idx_1],pred[idx_1])
    # print('group0:',TP_0, FP_0, TN_0, FN_0)
    # print('group1:',TP_1, FP_1, TN_1, FN_1)
    confusion = [( TP_0, FP_0, TN_0, FN_0),(TP_1, FP_1, TN_1, FN_1)]
    dp_gap = abs(pred_0.mean() - pred_1.mean())
    
    idx_00 = list(set(np.where(A_test==-1)[0]) & set(np.where(y==-1)[0]))
    idx_01 = list(set(np.where(A_test==-1)[0]) & set(np.where(y==1)[0]))
    idx_10 = list(set(np.where(A_test==1)[0]) & set(np.where(y==-1)[0]))
    idx_11 = list(set(np.where(A_test==1)[0]) & set(np.where(y==1)[0]))
    eo_tp_gap = pred[idx_11].mean() - pred[idx_01].mean()
    eo_fp_gap = (1-pred[idx_00]).mean() - (1-pred[idx_10]).mean()
    # print(len(idx_00),len(idx_01),len(idx_10),len(idx_11))
    eo_gap = (abs(eo_tp_gap) + abs(eo_fp_gap))/2

    
    acc = accuracy_score(y, pred)
    
    f1 = f1_score(y,pred)
    # print(f1)

    recall_0 = recall_score(y[idx_0], pred[idx_0])
    recall_1 = recall_score(y[idx_1], pred[idx_1])
    recall = recall_score(y,pred)
    
    return acc, dp_gap, eo_gap,f1, confusion, (recall, recall_0, recall_1)

def risk_estimator(args,loss_func,output,indicator):
    positive, unlabeled = indicator ==1, indicator == -1
    positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
    prior = args.prior.cuda()
    # print(prior)
    n_positive, n_unlabeled = torch.max(torch.tensor(1.), torch.sum(positive)), torch.max(torch.tensor(1.), torch.sum(unlabeled))
    y_positive = loss_func(positive*output) * positive
    y_positive_inv = loss_func(-positive*output) * positive
    y_unlabeled = loss_func(-unlabeled*output) * unlabeled
    positive_risk = prior * torch.sum(y_positive)/ n_positive
    negative_risk = - prior *torch.sum(y_positive_inv)/ n_positive + torch.sum(y_unlabeled)/n_unlabeled
    
    if negative_risk < -args.beta_pu and  args.pu_type =='nnpu':
        loss =  - args.gamma_pu * negative_risk
    else:
        
        loss =  positive_risk+negative_risk
    return loss





def make_pu(X, y, r1):
    s = np.ones((X.shape[0]))
    negative_idx = np.where(y==0)[0]
    positive_idx = np.where(y==1)[0]
    unlabeld_positive_idx = np.random.choice(positive_idx, size = int(len(positive_idx)*(r1)), replace=False) 
    s[negative_idx] = 0
    s[unlabeld_positive_idx] = 0
    return s


def lower_upper_loss(args, Y_train, output,sensitive, baselines):
    kappa = lambda z: torch.relu(1 + z)
    delta = lambda z: 1 - torch.relu(1 - z)
    N = len(output)
    
    y = Y_train.cpu().detach().numpy()
    sensitive = sensitive.cpu().detach().numpy()
    p0,p1,p11,p01,p10,p00= args.probs
    
    if args.fairness=='ddp':
        idx_0 = np.where(sensitive==-1)[0]
        idx_1 = np.where(sensitive==1)[0]
        pred_0 = output[idx_0]
        pred_1 = output[idx_1]
        pred_0 = torch.where(pred_0<0,1,0)
        pred_1 = torch.where(pred_1>=0,1,0)
        ddp = torch.sum(pred_1)/p1/N + torch.sum(pred_0)/p0/N -1
        
        if ddp.item()>0:
            wu_loss = args.lam_f * ((torch.sum(kappa(output[idx_1])/p1) + torch.sum(kappa(-output[idx_0])/p0))/N -1)
        else : 
            wu_loss = -1* args.lam_f * ((torch.sum(delta(output[idx_1])/p1)+ torch.sum(delta(-output[idx_0])/p0))/N -1 )
        if args.penalty:
            idx_00 = list(set(np.where(sensitive==-1)[0]) & set(np.where(y==-1)[0]))
            idx_01 = list(set(np.where(sensitive==-1)[0]) & set(np.where(y==1)[0]))
            idx_10 = list(set(np.where(sensitive==1)[0]) & set(np.where(y==-1)[0]))
            idx_11 = list(set(np.where(sensitive==1)[0]) & set(np.where(y==1)[0]))
            pred_00x = output[idx_00]
            pred_01x = output[idx_01]
            pred_10x = output[idx_10]
            pred_11x = output[idx_11]
            # Apply logistic surrogate function
            pred_00x = F.sigmoid(pred_00x)
            pred_01x = F.sigmoid(pred_01x)
            pred_10x = F.sigmoid(pred_10x)
            pred_11x = F.sigmoid(pred_11x)
            # print(pred_00x)
            # Group -1
            tpr0 = pred_01x.sum().float() / len(idx_01) if len(idx_01) > 0 else torch.tensor(0.0)
            fpr0 = pred_00x.sum().float() / len(idx_00) if len(idx_00) > 0 else torch.tensor(0.0)

            # Group 1
            tpr1 = pred_11x.sum().float() / len(idx_11) if len(idx_11) > 0 else torch.tensor(0.0)
            fpr1 = pred_10x.sum().float() / len(idx_10) if len(idx_10) > 0 else torch.tensor(0.0)
            
            if baselines is None:
                
                baselines = (tpr1.item(), tpr0.item(), fpr1.item(), fpr0.item())
            (tpr1_baseline, tpr0_baseline, fpr1_baseline, fpr0_baseline) = baselines
            
            tpr_penalty_1 = torch.max(torch.tensor(0.0), tpr1_baseline - tpr1)
            tpr_penalty_0 = torch.max(torch.tensor(0.0), tpr0_baseline - tpr0)
            fpr_penalty_1 = torch.max(torch.tensor(0.0), fpr1 - fpr1_baseline)
            fpr_penalty_0 = torch.max(torch.tensor(0.0), fpr0 - fpr0_baseline)
            penalty_loss = tpr_penalty_1 + tpr_penalty_0 + fpr_penalty_1 + fpr_penalty_0
            wu_loss+= args.lam_penalty * penalty_loss
            baselines = (max(tpr1.item(),tpr1_baseline), max(tpr0.item(),tpr0_baseline), min(fpr1.item(),fpr1_baseline), min(fpr0.item(),fpr0_baseline))
        return wu_loss ,baselines
    elif args.fairness=='deo':
        idx_00 = list(set(np.where(sensitive==-1)[0]) & set(np.where(y==-1)[0]))
        idx_01 = list(set(np.where(sensitive==-1)[0]) & set(np.where(y==1)[0]))
        idx_10 = list(set(np.where(sensitive==1)[0]) & set(np.where(y==-1)[0]))
        idx_11 = list(set(np.where(sensitive==1)[0]) & set(np.where(y==1)[0]))
        pred_00 = output[idx_00]
        pred_01 = output[idx_01]
        pred_10 = output[idx_10]
        pred_11 = output[idx_11]    
        
        pred_00 = torch.where(pred_00<0,1,0)
        pred_10 = torch.where(pred_10>=0,1,0)
        pred_01 = torch.where(pred_01<0,1,0)
        pred_11 = torch.where(pred_11>=0,1,0)
        deo = (torch.sum(pred_11)/p11 + torch.sum(pred_01)/p01 + torch.sum(pred_10)/p10 + torch.sum(pred_00)/p00)/N -2
        
        if deo.item()>=0:
            wu_loss = torch.sum(kappa(output[idx_11])/p11) + torch.sum(kappa(-output[idx_01])/p01) + torch.sum(kappa(output[idx_10])/p10) + torch.sum(kappa(-output[idx_00])/p00) 
            wu_loss = args.lam_f * (wu_loss/N - 2)
        else :
            wu_loss = torch.sum(delta(output[idx_11])/p11) + torch.sum(delta(-output[idx_01])/p01) + torch.sum(delta(output[idx_10])/p10) + torch.sum(delta(-output[idx_00])/p00) 
            wu_loss  =-1 * args.lam_f * (wu_loss/N - 2)

        if args.penalty:
            pred_00x = output[idx_00]
            pred_01x = output[idx_01]
            pred_10x = output[idx_10]
            pred_11x = output[idx_11]
            # Apply logistic surrogate function
            pred_00x = F.sigmoid(pred_00x)
            pred_01x = F.sigmoid(pred_01x)
            pred_10x = F.sigmoid(pred_10x)
            pred_11x = F.sigmoid(pred_11x)
            # print(pred_00x)
            # Group -1
            tpr0 = pred_01x.sum().float() / len(idx_01) if len(idx_01) > 0 else torch.tensor(0.0)
            fpr0 = pred_00x.sum().float() / len(idx_00) if len(idx_00) > 0 else torch.tensor(0.0)

            # Group 1
            tpr1 = pred_11x.sum().float() / len(idx_11) if len(idx_11) > 0 else torch.tensor(0.0)
            fpr1 = pred_10x.sum().float() / len(idx_10) if len(idx_10) > 0 else torch.tensor(0.0)
            
            if baselines is None:
                
                baselines = (tpr1.item(), tpr0.item(), fpr1.item(), fpr0.item())
            (tpr1_baseline, tpr0_baseline, fpr1_baseline, fpr0_baseline) = baselines
            
            tpr_penalty_1 = torch.max(torch.tensor(0.0), tpr1_baseline - tpr1)
            tpr_penalty_0 = torch.max(torch.tensor(0.0), tpr0_baseline - tpr0)
            fpr_penalty_1 = torch.max(torch.tensor(0.0), fpr1 - fpr1_baseline)
            fpr_penalty_0 = torch.max(torch.tensor(0.0), fpr0 - fpr0_baseline)
            penalty_loss = tpr_penalty_1 + tpr_penalty_0 + fpr_penalty_1 + fpr_penalty_0
            wu_loss+= args.lam_penalty * penalty_loss
            baselines = (max(tpr1.item(),tpr1_baseline), max(tpr0.item(),tpr0_baseline), min(fpr1.item(),fpr1_baseline), min(fpr0.item(),fpr0_baseline))
            
        
        return wu_loss,baselines
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True