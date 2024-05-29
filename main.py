import argparse
from data.preprocess import SETGENERATOR
import warnings
warnings.filterwarnings("ignore")
import os
from utils.config import LOGTYPE,COL,META,TASK,INJM
from operator import itemgetter
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
random.seed(42)
from models.model import MODEL,Train,TestClassification,TestRegression
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

import json
import h5py

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Dataset Generator")


parser.add_argument("--dataset_path", type=str,
                    default='data/datasets/',
                    help=" data location ")

parser.add_argument("--dataset_name", required=True, type=str,
                    choices=["BPIC15_1_f2", "BPIC11_f1", "Credit","Pub"],
                    help="dataset name (one of: BPIC15_1_f2, BPIC11_f1, Credit, Pub)")


parser.add_argument("--trials",  type=int,
                    default= 4,
                    help=" test trials ")

parser.add_argument("--task", required=True, type=str,
                    choices=["next_activity", "event_remaining_time","outcome", "case_remaining_time"],
                    help="task name (one of: next_activity, outcome, event_remaining_time, case_remaining_time)")

parser.add_argument("--modelpath", type=str, 
    default='models/model.pt',  help="model location")

parser.add_argument("--batchsize", type=int, 
    default=16,  help="batchsize for input")

parser.add_argument("--lr", type=float, 
    default=0.001,  help="learning rate")

parser.add_argument("--hiddendim", type=int, 
    default=128,  help="size of hidden layer")

parser.add_argument("--embdim", type=int, 
    default=32,  help="size of embedding layer")

parser.add_argument("--epoch", type=int, 
    default=300,  help="number of epoch")

parser.add_argument("--result_path", type=str,
                    default='result/',
                    help=" save location ")


args = parser.parse_args()


def interval_accuracy(true_bin,pred_bin,target_bin):
    df = pd.DataFrame({
        'true': true_bin,
        'pred': pred_bin,
        'target': target_bin,
        })
    # Divide ratio into intervals (e.g., 0~0.1, 0.1~0.2, ..., 0.9~1.0)
    target_bins = pd.cut(df['target'], bins=[i/10 for i in range(11)], include_lowest=True)
    df['target_bin'] = target_bins

    # Calculate accuracy for each injection ratio or sequence length ratio interval
    each_accuracy = df.groupby('target_bin').apply(lambda x: accuracy_score(x['true'], x['pred']))
    each_accuracy = each_accuracy.reset_index().rename(columns={0: 'accuracy'})
    return each_accuracy


if __name__ == "__main__": 
    if args.dataset_name in ["BPIC15_1_f2", "BPIC11_f1"]:
        pattern_bin = ["CLEAN","DISTORTED", "POLLUTED.NORND", "POLLUTED.RANDOM"]
    else : 
        pattern_bin = ["CLEAN","DISTORTED", "POLLUTED.NORND", "POLLUTED.RANDOM", "SYNONYM", "HOMONYM"]
    inj_ratio_bin = ['0.1','0.2','0.3','0.4','0.5']
    combinations = []
    for p1 in pattern_bin:
        if p1 == "CLEAN":
            combinations.append([f"-TRAIN-{p1}", f"-TEST-{p1}",p1,p1])
            for p2 in pattern_bin:
                if p2 != "CLEAN":
                    for ratio in inj_ratio_bin:
                        combinations.append([f"-TRAIN-{p1}", f"-TEST-{p2}-{ratio}",p1,p2])
                        combinations.append([f"-TRAIN-{p2}-{ratio}", f"-TEST-{p1}",p2,p1])
    for p1 in pattern_bin:
        if p1 != "CLEAN":
            for ratio1, ratio2 in product(inj_ratio_bin, repeat=2):
                combinations.append([f"-TRAIN-{p1}-{ratio1}",f"-TEST-{p1}-{ratio2}",p1,p1])

    for comb in combinations:
        true_bin,pred_bin,length_bin,ratio_bin = [],[],[],[]
        for rep in range(args.trials):
            if "CLEAN" in comb[0]:
                train_csv = f"{args.dataset_path}{args.dataset_name}{comb[0]}.csv"
            else:
                train_csv = f"{args.dataset_path}{args.dataset_name}{comb[0]}-{rep}.csv"
                
            if "CLEAN" in comb[1]:
                test_csv = f"{args.dataset_path}{args.dataset_name}{comb[1]}.csv" 
            else:
                test_csv = f"{args.dataset_path}{args.dataset_name}{comb[1]}-{rep}.csv"
                
            print("Train_csv : ", train_csv,'\n',"Test_csv : ",  test_csv, '\n',"Replication # : ", rep)
    
            set_generator = SETGENERATOR(train_csv, test_csv, comb[2],comb[3], args.task, args.batchsize)
            train_loader, test_loader, meta = set_generator.SetGenerator()
            output_dim, num_act, max_length, attr_size, scaler = itemgetter(META.OUTDIM.value, META.NUMACT.value, META.MAXLEN.value, META.ATTRSZ.value,META.SCALER.value)(meta)
            
            model = MODEL(num_act, args.embdim, args.hiddendim, 2, attr_size, output_dim, args.task).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            if args.task == TASK.NAP.value:
                criterion = nn.CrossEntropyLoss()
            elif args.task == TASK.OP.value:
                criterion = nn.BCEWithLogitsLoss()
            else: 
                criterion = nn.SmoothL1Loss()
            
            Train(model, train_loader, criterion, optimizer, args.epoch, args.modelpath, args.task)
            
            model.load_state_dict(torch.load(args.modelpath))
            if (args.task == TASK.NAP.value) or (args.task == TASK.OP.value):
                true,pred,length,ratio = TestClassification(model, test_loader, output_dim, args.task, comb[3])
            else:
                true,pred,length,ratio = TestRegression(model, test_loader, scaler, comb[3])
                
            true_bin.extend(true),pred_bin.extend(pred),length_bin.extend(length),ratio_bin.extend(ratio)




        overall_accuracy = accuracy_score(true_bin, pred_bin)
        overall_fscore = f1_score(true_bin, pred_bin,average='macro')


        results_summary = {
            'overall_accuracy': overall_accuracy,
            'overall_fscore': overall_fscore
        }

        with open(f'{args.result_path}{args.dataset_name}-{args.task}{comb[0]}-{comb[1]}-overall.json', 'w') as f:
            json.dump(results_summary, f)       
        
        
        length_accuracy  = interval_accuracy(true_bin,pred_bin, [x / max(length_bin) for x in length_bin])
        length_accuracy.to_csv(f'{args.result_path}{args.dataset_name}-{args.task}{comb[0]}-{comb[1]}-length_accuracy.csv', index=False)

        if comb[3] != 'CLEAN':
            ratio_accuracy = interval_accuracy(true_bin,pred_bin, ratio_bin)
            ratio_accuracy.to_csv(f'{args.result_path}{args.dataset_name}-{args.task}{comb[0]}-{comb[1]}-ratio_accuracy.csv', index=False)  
        
        
            
