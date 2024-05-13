import os
import random
import numpy as np
import json
from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle
from utils.config import TASK,NAME
random.seed(42)

class SETLOADER:
    def __init__(self, dataset, meta, filepath, task, batchsize):
        self._dataset = dataset
        self._meta = meta
        self._filepath = filepath
        self._task = task
        self._batchsize = batchsize
        if not os.path.exists(f"{self._filepath}{self._dataset[:-4]}/{self._task}/{self._dataset[:-4]}_train.npz"):
            raise FileNotFoundError("The train set does not exist.")
            
        if not os.path.exists(f"{self._filepath}{self._dataset[:-4]}/{self._task}/{self._dataset[:-4]}_test.npz"):
            raise FileNotFoundError("The test set does not exist.")

    def _labeltester(self, label_array)->None:
        for i in range(label_array.shape[0]):
            if not np.all(label_array[i, :, :] == label_array[i, 0, :]):
                raise ValueError("Label data polluted")
        return None

    def _setloader(self,train_array,test_array)-> Tuple[DataLoader,DataLoader]:
        train_act = torch.tensor(train_array[NAME.ACT.value], dtype=torch.int64)
        train_attr = torch.tensor(train_array[NAME.ATTR.value], dtype=torch.float)
        test_act = torch.tensor(test_array[NAME.ACT.value], dtype=torch.int64)
        test_attr = torch.tensor(test_array[NAME.ATTR.value], dtype=torch.float)
        
        train_label = train_array[NAME.LABEL.value]
        test_label = test_array[NAME.LABEL.value]
        self._labeltester(train_label)
        self._labeltester(test_label)
        
        if (self._task == TASK.NAP.value) or (self._task == TASK.OP.value):
            train_label = torch.tensor(train_label[:, 0:1, :], dtype=torch.int64).squeeze()
            test_label = torch.tensor(test_label[:, 0:1, :], dtype=torch.int64).squeeze()
        else:
            train_label = torch.tensor(train_label[:, 0:1, :], dtype=torch.float).squeeze()
            test_label = torch.tensor(test_label[:, 0:1, :], dtype=torch.float).squeeze()
        train_tensor = TensorDataset(train_act,train_attr,train_label)
        test_tensor = TensorDataset(test_act,test_attr,test_label)
        
        train_set = DataLoader(train_tensor, batch_size=self._batchsize, shuffle=True)
        test_set = DataLoader(test_tensor, batch_size=self._batchsize, shuffle=False)
        return train_set,test_set


    def SetLoader(self)-> Tuple[DataLoader, DataLoader, dict, MinMaxScaler]:
        with open(f"{self._filepath}{self._dataset[:-4]}/{self._task}/{self._meta}", 'r') as json_file:
            meta_dict = json.load(json_file)
        if (self._task == TASK.NAP.value) or (self._task == TASK.OP.value):
            scaler = None
        else:
            with open(f"{self._filepath}{self._dataset[:-4]}/{self._task}/scaler.pkl", 'rb') as file:
                scaler = pickle.load(file)
            
        train_array = np.load(f"{self._filepath}{self._dataset[:-4]}/{self._task}/{self._dataset[:-4]}_train.npz")
        test_array = np.load(f"{self._filepath}{self._dataset[:-4]}/{self._task}/{self._dataset[:-4]}_test.npz")
        train_set, test_set = self._setloader(train_array,test_array)
        return train_set,test_set,meta_dict,scaler
    
    
