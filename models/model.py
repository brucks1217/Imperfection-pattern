import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys

from utils.config import TASK,INJM
import random
import numpy as np
random.seed(42)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MODEL(nn.Module):
    def __init__(self, num_act, embedding_dim, hidden_dim, num_layers, attr_size, output_dim, task):
        super(MODEL, self).__init__()
        self.embedding = nn.Embedding(num_act, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.main_layers = nn.ModuleList()
        for i in range(num_layers):
            self.main_layers.append(nn.LSTM(embedding_dim + attr_size if i == 0 else hidden_dim, hidden_dim, batch_first=True))
        self.fc = nn.Linear(hidden_dim, output_dim)
        for mlayer in self.main_layers:
            if isinstance(mlayer, nn.LSTM):
                for name, param in mlayer.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
    def forward(self, act, attr, lengths):
        embedded = self.embedding(act)
        combined_features = torch.cat((embedded, attr), dim=2)
        packed_input = pack_padded_sequence(combined_features, lengths, batch_first=True, enforce_sorted=True)
        output = packed_input
        for lstm in self.main_layers:
            output, _ = lstm(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.fc(output)
        return output
    
 
def Train(model, train_loader, criterion, optimizer, num_epochs, model_path,task):
    loss_ = np.Inf
    early_count = 0
    loss_history = []  
    display_interval = 5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    for epoch in range(num_epochs):
        for i, (act, attr, label, lengths) in enumerate(train_loader):
            #act, attr = act.to(device), attr.to(device)
            act, attr, label = act.to(device), attr.to(device), label[:,:lengths[0]].to(device)
            outputs = model(act,attr, lengths)
            if task == TASK.NAP.value:
                outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * sequence_length, output_dim)
                label = label.view(-1)  # (batch_size * sequence_length)
                loss = criterion(outputs, label)
            else:
                outputs = outputs.squeeze(-1)  # (batch_size, sequence_length)
                label = label.float()  # (batch_size, sequence_length)
                loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_history.append(loss.item())

        if (epoch + 1) % display_interval == 0:
            history_str = " | ".join(f"{l:.6f}" for l in loss_history[-display_interval:])
            sys.stdout.write(f'\rEpoch [{epoch+1}/{num_epochs}], Current Best Model Loss: {loss_:.6f}, Previous 5 Epochs Losses: {history_str}')
            sys.stdout.flush()
                  
        if (loss.item() + 0.0001)< loss_:
            loss_ = loss.item()
            torch.save(model.state_dict(), model_path)
            early_count = 0
        else:
            if (early_count > 9) and (epoch > 100):
                break
            early_count += 1
        if epoch > 100:
            scheduler.step()

    print(f"\nSaved Model Loss : {loss_}")



def TestClassification(model, test_loader, output_dim, task, injection_type):
    model.eval()  
    true,pred = [], []
    length_bin, ratio_bin = [],[]
    if injection_type == INJM.CLN.value:
        with torch.no_grad():
            for act, attr, label, lengths in test_loader:
                act, attr, label = act.to(device), attr.to(device), label.to(device)
                outputs = model(act, attr, lengths)
                
                
                indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, outputs.size(2)).to(torch.int64).to(device)
                outputs = outputs.gather(1, indices).squeeze(1)
                indices = (lengths - 1).unsqueeze(1).to(torch.int64).to(device)  # (batch_size, 1)
                label = label.gather(1, indices).squeeze(1)
                #print(outputs.size(),label.size())
                if task == TASK.NAP.value:
                    _, predicted = torch.max(outputs.data, 1)
                    pred.extend(predicted.cpu().numpy())
                elif task == TASK.OP.value:
                    predicted = torch.sigmoid(outputs).round().squeeze().tolist()
                    pred.extend(predicted)
                true.extend(label.cpu().numpy())
                length_bin.extend(lengths.cpu().numpy())
                
                
    else:
        with torch.no_grad():
            for act, attr, label, lengths, ratio in test_loader:
                act, attr, label = act.to(device), attr.to(device), label.to(device)
                outputs = model(act, attr, lengths)

                indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, outputs.size(2)).to(torch.int64).to(device)
                outputs = outputs.gather(1, indices).squeeze(1)
                indices = (lengths - 1).unsqueeze(1).to(torch.int64).to(device)  # (batch_size, 1)
                label = label.gather(1, indices).squeeze(1)


                if task == TASK.NAP.value:
                    _, predicted = torch.max(outputs.data, 1)
                    pred.extend(predicted.cpu().numpy())
                elif task == TASK.OP.value:
                    predicted = torch.sigmoid(outputs).round().squeeze().tolist()
                    pred.extend(predicted)
                true.extend(label.cpu().numpy())
                length_bin.extend(lengths.cpu().numpy())
                ratio_bin.extend(ratio.cpu().numpy())
    return true,pred,length_bin,ratio_bin


def TestRegression(model, test_loader, scaler , injection_type):
    model.eval()  
    pred = []
    true = []
    length_bin, ratio_bin = [],[]

    if injection_type == INJM.CLN.value:
        with torch.no_grad():  
            for act, attr, label, lengths in test_loader:
                act, attr, label = act.to(device), attr.to(device), label.to(device)
                outputs = model(act, attr, lengths)
                
                indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, outputs.size(2)).to(torch.int64).to(device)
                outputs = outputs.gather(1, indices).squeeze(1)
                indices = (lengths - 1).unsqueeze(1).to(torch.int64).to(device)  # (batch_size, 1)
                label = label.gather(1, indices).squeeze(1)
                
                
                outputs = outputs.view(-1).cpu().numpy()
                outputs = scaler.inverse_transform(outputs.reshape(-1, 1)).flatten()
                label = label.view(-1).cpu().numpy()
                label = scaler.inverse_transform(label.reshape(-1, 1)).flatten() # Time UNIT : second 
                pred.extend(outputs)
                true.extend(label)
                length_bin.extend(lengths.numpy())
    else:
        with torch.no_grad():  
            for act, attr, label, lengths, ratio in test_loader:
                act, attr, label = act.to(device), attr.to(device), label.to(device)
                outputs = model(act, attr, lengths)
                
                indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, outputs.size(2)).to(torch.int64).to(device)
                outputs = outputs.gather(1, indices).squeeze(1)
                indices = (lengths - 1).unsqueeze(1).to(torch.int64).to(device)  # (batch_size, 1)
                label = label.gather(1, indices).squeeze(1)
                
                outputs = outputs.view(-1).cpu().numpy()
                outputs = scaler.inverse_transform(outputs.reshape(-1, 1)).flatten()
                label = label.view(-1).cpu().numpy()
                label = scaler.inverse_transform(label.reshape(-1, 1)).flatten() # Time UNIT : second 
                pred.extend(outputs)
                true.extend(label)
                length_bin.extend(lengths.numpy())
                ratio_bin.extend(ratio.numpy())
        
    #mse = mean_squared_error(true, pred)
   # mae = mean_absolute_error(true, pred)
   # r2 = r2_score(true, pred)
   # rmse = np.sqrt(mse)
    
   # metrics = {'MSE': mse,'MAE': mae,'R2': r2,'RMSE': rmse    }

    return true,pred,length_bin,ratio_bin
