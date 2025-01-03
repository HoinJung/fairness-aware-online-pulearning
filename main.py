import argparse
import os
import numpy as np
from train import run_train
from online_train import run_online_train
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

                
def main(args):
    if args.online : 
        print(f'Run Online training {args.dataset} {args.model} {args.pu_type}'.upper())
        run_online_train(args)
        
    else : 
        print(f'Run training {args.dataset} {args.model} {args.pu_type}'.upper())
        run_train(args)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mlp', type=str, help='mlp or linear')
    parser.add_argument('--dataset', default='chat_toxicity', type=str,help='nela, wiki, chat_toxicity')
    parser.add_argument('--gpu_id', default='1', type=str)
    parser.add_argument('--r1', default=0.5, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--num_exp', default=10, type=int)
    parser.add_argument('--pu_type', default='nnpu', type=str)
    parser.add_argument('--opt', default='adam', type=str)
    parser.add_argument('--loss_type', default='dh', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--round', default=200, type=int)
    parser.add_argument('--b', default=20, type=int)
    parser.add_argument('--fairness', default='none', type=str, help='ddp or deo')
    
    parser.add_argument('--online',default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--fair',default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--verbose',default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--penalty',default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lam_f', default=1, type=float)
    parser.add_argument('--lam_penalty', default=1, type=float)
    parser.add_argument('--lam', default=0.1, type=float)
    parser.add_argument('--num_layers', type=int, default=12)
    
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    main(args)