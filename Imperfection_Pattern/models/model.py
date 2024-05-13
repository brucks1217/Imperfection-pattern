import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support,accuracy_score,mean_squared_error, mean_absolute_error, r2_score
from utils.config import TASK
import random
import numpy as np
random.seed(42)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MODEL(nn.Module):
    def __init__(self, num_act, embedding_dim, hidden_dim, num_layers, attr_size, output_dim, task):
        super(MODEL, self).__init__()

        self.embedding = nn.Embedding(num_act, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.main_layers = nn.ModuleList()
        for i in range(num_layers):
            self.main_layers.append(nn.LSTM(embedding_dim + attr_size if i == 0 else hidden_dim, hidden_dim, batch_first=True))
        
        
        #if task == TASK.NAP.value:
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
        for mlayer in self.main_layers:
            if isinstance(mlayer, nn.LSTM):
                for name, param in mlayer.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)


    def forward(self, act, attr):
        embedded = self.embedding(act.squeeze(-1))
        combined_features = torch.cat((embedded, attr), dim=2)
        output = combined_features
        for lstm in self.main_layers:
            output, _ = lstm(output)
        output = output[:, -1, :]
        output = self.fc(output)
        return output
    
 
def Train(model, train_loader, criterion, optimizer, num_epochs, model_path,task):
    loss_ = np.Inf
    early_count = 0
    for epoch in range(num_epochs):
        for i, (act, attr, label) in enumerate(train_loader):
            act, attr, label = act.to(device), attr.to(device), label.to(device)
            outputs = model(act,attr)
            if task == TASK.NAP.value:
                loss = criterion(outputs, label)
            else:
                loss = criterion(outputs.squeeze(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')      
                  
        if loss.item() < loss_:
            loss_ = loss.item()
            torch.save(model.state_dict(), model_path)
            early_count = 0
        else:
            if early_count > 5:
                break
            early_count += 1



def TestClassification(model, test_loader, output_dim, task):
    pred_auc = np.empty((0, output_dim))
    true_auc = np.array([])
    model.eval()  
    all_labels,all_preds = [], []

    with torch.no_grad():
        for act, attr, label in test_loader:
            act, attr, label = act.to(device), attr.to(device), label.to(device)
            outputs = model(act, attr)
            
            if task == TASK.NAP.value:
                _, predicted = torch.max(outputs.data, 1)
                pred_auc = np.vstack((pred_auc, nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()))
                all_preds.extend(predicted.cpu().numpy())
            elif task == TASK.OP.value:
                predicted = torch.sigmoid(outputs).round().squeeze().tolist()
                pred_auc = np.vstack((pred_auc, torch.sigmoid(outputs).cpu().detach().numpy()))
                all_preds.extend(predicted)
            all_labels.extend(label.cpu().numpy())
            
            if true_auc.size == 0:
                true_auc = label.to('cpu').numpy()
            else:
                true_auc = np.concatenate((true_auc, label.to('cpu').numpy()))
                
    precision, recall, fscore, support = precision_recall_fscore_support(all_labels, all_preds, average='macro')    
    metrics = {
        'accuracy score' : accuracy_score(all_labels, all_preds),
        'roc_auc_score' : roc_auc_score(true_auc, pred_auc, multi_class='ovo', labels=list(range(output_dim))),
        'f score' : fscore
    }
    return  metrics

def TestRegression(model, test_loader, scaler):
    model.eval()  
    pred = []
    true = []
    
    with torch.no_grad():  
        for act, attr, label in test_loader:
            act, attr, label = act.to(device), attr.to(device), label.to(device)
            outputs = model(act, attr)
            
            outputs = outputs.view(-1).cpu().numpy()
            outputs = scaler.inverse_transform(outputs.reshape(-1, 1)).flatten()
            label = label.view(-1).cpu().numpy()
            label = scaler.inverse_transform(label.reshape(-1, 1)).flatten() # Time UNIT : second 
            pred.extend(outputs)
            true.extend(label)
            #pred.extend(outputs.view(-1).cpu().numpy())
            #true.extend(label.view(-1).cpu().numpy())
            
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse
    }

    return metrics