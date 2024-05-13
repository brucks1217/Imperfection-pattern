import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from data.loader import SETLOADER
from utils.config import TASK,NAME
from operator import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
random.seed(42)
from models.model import MODEL,Train,TestClassification,TestRegression
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



parser = argparse.ArgumentParser(description="Prefix + Zero padd -> LSTM -> Outcome/NextAct/EventRemainTime/CaseRemainTime")

parser.add_argument("--dataset", required=True, type=str,
                    choices=["BPIC11.csv", "BPIC15_1.csv", "credit-card-new.csv", "pub-new.csv"],help="event log name")

parser.add_argument("--meta",  type=str,
                    default='meta_info.json',help="meta dict name")

parser.add_argument("--filepath", type=str, 
    default='data/processed/',  help="dataset location")

parser.add_argument("--task", required=True, type=str,
                    choices=["next_activity", "outcome", "event_remaining_time", "case_remaining_time"],
                    help="task name (one of: next_activity, outcome, event_remaining_time, case_remaining_time)")

parser.add_argument("--modelpath", type=str, 
    default='models/model.pt',  help="model location")

parser.add_argument("--batchsize", type=int, 
    default=32,  help="batchsize for input")

parser.add_argument("--lr", type=float, 
    default=0.001,  help="learning rate")

parser.add_argument("--hiddendim", type=int, 
    default=128,  help="size of hidden layer")

parser.add_argument("--embdim", type=int, 
    default=32,  help="size of embedding layer")

parser.add_argument("--epoch", type=int, 
    default=100,  help="number of epoch")
args = parser.parse_args()
 
if __name__ == "__main__":  # args.dataset, args.meta, args.filepath, args.task, args.modelpath, args.batchsize, args.lr, args.hiddendim, args.embdim, args.epoch 
    set_generator = SETLOADER(args.dataset, args.meta, args.filepath, args.task, args.batchsize)
    train_set, test_set, meta_dict, scaler = set_generator.SetLoader() # in case of BPIC15: number of batch = train 38887/ test 12131. size of act : [32,100,1], attr : [32,100,137], label: [32,100,1]
    output_dim, num_act, max_length, attr_size = itemgetter(NAME.OUTDIM.value, NAME.NUMACT.value, NAME.MAXLEN.value, NAME.ATTRSZ.value)(meta_dict)
    
    model = MODEL(num_act, args.embdim, args.hiddendim, 2, attr_size, output_dim, args.task).to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.task == TASK.NAP.value:
        criterion = nn.CrossEntropyLoss()
    elif args.task == TASK.OP.value:
        criterion = nn.BCEWithLogitsLoss()
    else: # next event/case remain time
        criterion = nn.MSELoss()
        
    Train(model, train_set, criterion, optimizer, args.epoch, args.modelpath, args.task)
    
    model.load_state_dict(torch.load(args.modelpath))
    if (args.task == TASK.NAP.value) or (args.task == TASK.OP.value):
        metrics = TestClassification(model, test_set, output_dim, args.task)
        print(metrics)
    else:
        metrics = TestRegression(model, test_set, scaler)
        print(metrics)