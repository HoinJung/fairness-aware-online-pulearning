import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import torch
import copy
def evaluate(output, y_test, A_test):

    y = copy.deepcopy(y_test)
    
    active=(lambda x : torch.sigmoid(x))
    output=output[:,0]
    idx_0 = np.where(A_test==0)[0]
    idx_1 = np.where(A_test==1)[0]
    pred_0 = active(output[idx_0])
    pred_1 = active(output[idx_1])
    
    dp_gap = pred_0.mean() - pred_1.mean()
    dp_gap = abs(dp_gap.data.cpu().numpy())
    
    # Equalized odds gap
    idx_00 = list(set(np.where(A_test==0)[0]) & set(np.where(y==-1)[0]))
    idx_01 = list(set(np.where(A_test==0)[0]) & set(np.where(y==1)[0]))
    idx_10 = list(set(np.where(A_test==1)[0]) & set(np.where(y==-1)[0]))
    idx_11 = list(set(np.where(A_test==1)[0]) & set(np.where(y==1)[0]))

    pred_00 = active(output[idx_00])
    pred_01 = active(output[idx_01])
    pred_10 = active(output[idx_10])
    pred_11 = active(output[idx_11])

    gap_0 = pred_00.mean() - pred_10.mean()
    gap_1 = pred_01.mean() - pred_11.mean()
    gap_0 = abs(gap_0.data.cpu().numpy())
    gap_1 = abs(gap_1.data.cpu().numpy())

    eo_gap = gap_0 + gap_1

    y_scores = output.data.cpu().numpy()
    threshold = 0
    y_pred = np.where(y_scores>=threshold, 1, -1)
    # f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)


    return acc, dp_gap, eo_gap

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


def lower_upper_loss(args, Y_train, output,sensitive):
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
        return wu_loss 
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
        
        return wu_loss
