import os
import pickle
import random
import numpy as np
import json
from collections import Counter

from sklearn.preprocessing import StandardScaler
from typing import Tuple
import pandas as pd
import torch

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader

random.seed(42)


from utils.config import LOGTYPE,COL,META,TASK,INJM
class SETGENERATOR:
    def __init__(self, train_csv, test_csv, inj_type_train, inj_type_test, task, batchsize):
        self._train_csv = train_csv
        self._test_csv = test_csv
        self._task = task
        self._batchsize= batchsize
        self._inj_type_train = inj_type_train
        self._inj_type_test = inj_type_test
        self._logtype_train = LOGTYPE(self._inj_type_train,self._task)
        self._logtype_test = LOGTYPE(self._inj_type_test,self._task)
        
    def _log_reader(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        df_train = pd.read_csv(self._train_csv,
                               usecols=list(self._logtype_train.keys()),
                               dtype={k: v for k, v in self._logtype_train.items() if k != COL.TIME.value},
                               parse_dates=[COL.TIME.value])
        
        df_test = pd.read_csv(self._test_csv,
                               usecols=list(self._logtype_test.keys()),
                               dtype={k: v for k, v in self._logtype_test.items() if k != COL.TIME.value},
                               parse_dates=[COL.TIME.value])        
        return df_train, df_test
    
    def _tokenization(self,df_train, df_test, column) -> Tuple[pd.DataFrame, pd.DataFrame]:   
        df_train[column].unique() 
        unique_act = list(set(df_train[column].unique() ).union(set(df_test[column].unique() )))
        str_to_idx = {a: i+1 for i, a in enumerate(unique_act)}
        df_train[column] = df_train[column].map(str_to_idx)
        df_test[column] = df_test[column].map(str_to_idx)     
        return df_train,df_test

    def _time_feature(self,df) ->pd.DataFrame:
        df[COL.TSP.value] = (df[COL.TIME.value] - df.groupby(COL.CASE.value)[COL.TIME.value].shift(1)).dt.total_seconds()
        df[COL.TSP.value] = df[COL.TSP.value].fillna(0)
        
        df[COL.TSSC.value] = (df[COL.TIME.value] - df.groupby(COL.CASE.value)[COL.TIME.value].transform('min')).dt.total_seconds()
        return df

    
    def _nap_label(self,df_train,df_test) -> Tuple[pd.DataFrame,pd.DataFrame,int]:

        df_train[COL.LABEL.value] = df_train.groupby(COL.CASE.value)[COL.ACT.value].shift(-1)
        df_test[COL.LABEL.value] = df_test.groupby(COL.CASE.value)[COL.ACT.value].shift(-1)
        max_idx = max([df_train[COL.LABEL.value].max(),df_test[COL.LABEL.value].max()])
        
        df_train[COL.LABEL.value] = df_train[COL.LABEL.value].fillna(max_idx+1).astype(int)
        #df_test[COL.LABEL.value] = df_test[COL.LABEL.value].fillna('finished').astype(str)
        
        df_test[COL.LABEL.value] = df_test[COL.LABEL.value].fillna(max_idx+1).astype(int)
        unique_label = list(set(df_train[COL.LABEL.value].unique() ).union(set(df_test[COL.LABEL.value].unique() )))
        lab_to_idx = {a: i for i, a in enumerate(unique_label)}
        df_train[COL.LABEL.value] = df_train[COL.LABEL.value].map(lab_to_idx)
        df_test[COL.LABEL.value] = df_test[COL.LABEL.value].map(lab_to_idx)  
              
        output_dim = len(unique_label)
        return df_train, df_test, output_dim
    
    def _op_label(self,df_train,df_test) -> pd.DataFrame:   
        df_train[COL.LABEL.value] = df_train[COL.OC.value].map({'deviant': 0, 'regular': 1})
        df_test[COL.LABEL.value] = df_test[COL.OC.value].map({'deviant': 0, 'regular': 1})
        return df_train,df_test
    
    def _ertp_label(self,df_train,df_test) -> Tuple[pd.DataFrame,pd.DataFrame]:
        df_train[COL.LABEL.value] = df_train.groupby(COL.CASE.value)[COL.TIME.value].shift(-1) - df_train[COL.TIME.value]
        df_train[COL.LABEL.value] = df_train[COL.LABEL.value].dt.total_seconds()
        df_train[COL.LABEL.value] = df_train[COL.LABEL.value].fillna(0).astype(float)
        
        df_test[COL.LABEL.value] = df_test.groupby(COL.CASE.value)[COL.TIME.value].shift(-1) - df_test[COL.TIME.value]
        df_test[COL.LABEL.value] = df_test[COL.LABEL.value].dt.total_seconds()        
        df_test[COL.LABEL.value] = df_test[COL.LABEL.value].fillna(0).astype(float)
        return df_train,df_test
    
    def _crtp_label(self,df_train,df_test) -> Tuple[pd.DataFrame,pd.DataFrame]:
        temp = df_train.groupby(COL.CASE.value)[COL.TIME.value].transform('max')
        df_train[COL.LABEL.value] = (temp- df_train[COL.TIME.value]).dt.total_seconds()
        
        temp = df_test.groupby(COL.CASE.value)[COL.TIME.value].transform('max')
        df_test[COL.LABEL.value] = (temp- df_test[COL.TIME.value]).dt.total_seconds()
        return df_train,df_test
    
    
    def _scaler(self,df_train, df_test, column) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        scaler = StandardScaler()
        df_train[column] = scaler.fit_transform(df_train[[column]])
        df_test[column] = scaler.transform(df_test[[column]])

        return df_train, df_test, scaler
        
    
    def _meta_scrap(self,df_train,df_test) :
        num_act = len(list(set(df_train[COL.ACT.value].unique()).union(set(df_test[COL.ACT.value].unique())))) 
        num_res = len(list(set(df_train[COL.RES.value].unique()).union(set(df_test[COL.RES.value].unique())))) 
        max_length = max(df_train.groupby(COL.CASE.value).size().max(),df_test.groupby(COL.CASE.value).size().max())
        if COL.INJ.value in df_test.columns:
            inject_act_list = list(set(df_test[df_test[COL.INJ.value].notna()][COL.ACT.value]))
        else:
            inject_act_list = None

        return num_act, num_res, max_length, inject_act_list
    
    def _zero_pad(self,trace_tensor, max_length,for_train ,inject_act_list=None):
        prefix_bin = []
        length_bin = []
        inj_ratio_list = []
        for prfx in range(1,len(trace_tensor)+1):
            prefixed_tensor = trace_tensor[:prfx]
            length_bin.append(torch.tensor(prfx, dtype=torch.int))
            if inject_act_list is not None:
                temp_act = prefixed_tensor.view(-1).tolist()
                inj_counter = Counter(temp_act)
                inj_count = sum(inj_counter[value] for value in inject_act_list)
                tensor_length = len(temp_act)
                inj_ratio_list.append(inj_count / tensor_length if inj_count > 0 else 0)
            
            
            pad_size = max_length - prefixed_tensor.size(0)
            prefixed_tensor = F.pad(prefixed_tensor, (0, 0, 0, pad_size), value= 0.0)
            prefix_bin.append(prefixed_tensor)
        
        padded_trace = torch.stack(prefix_bin, dim=0)
        inj_ratio_list = torch.tensor(inj_ratio_list)
        length_bin = torch.stack(length_bin).view(-1, 1)
        if for_train:
            padded_trace = padded_trace[-1,:,:].unsqueeze(0)
            length_bin = length_bin[-1,:].unsqueeze(0)
        if inject_act_list is not None: 
            return padded_trace, inj_ratio_list
        else:
            return padded_trace, length_bin
        
        
            
    def _prfx_bucket(self,df,num_res,max_length,for_train,inject_act_list=None) :
        cases = df.groupby(COL.CASE.value)
        inputs_act = []
        inputs_attr = []
        inputs_label = []
        inputs_length = []
        meta_ratio = []
        
        for _, trace in cases:
            trace_attr = torch.tensor(trace[[COL.TSP.value, COL.TSSC.value]].values, dtype=torch.float)
            resource_oh = F.one_hot(torch.tensor(trace[COL.RES.value].values-1), num_classes=num_res).float()
            trace_attr = torch.cat([resource_oh, trace_attr], dim=1)
            trace_act = torch.tensor(trace[COL.ACT.value].values, dtype=torch.int64).unsqueeze(1)
            if (self._task == TASK.NAP.value) or (self._task ==TASK.OP.value):
                trace_label = torch.tensor(trace[COL.LABEL.value].values, dtype=torch.int64).unsqueeze(1)
            else:
                trace_label = torch.tensor(trace[COL.LABEL.value].values, dtype=torch.float).unsqueeze(1)

            if trace_attr.size(0) != trace_act.size(0) or trace_attr.size(0) != trace_label.size(0) or trace_act.size(0) != trace_label.size(0):
                raise ValueError(f"Size mismatch: trace_attr.size(0)={trace_attr.size(0)}, trace_act.size(0)={trace_act.size(0)}, trace_label.size(0)={trace_label.size(0)}")

            
            zero_attr,_ = self._zero_pad(trace_attr,max_length,for_train)
            zero_label,zero_length = self._zero_pad(trace_label,max_length,for_train)
            inputs_attr.append(zero_attr)
            inputs_label.append(zero_label)
            inputs_length.append(zero_length)
            
            if inject_act_list is not None:
                zero_act, inj_ratio_list = self._zero_pad(trace_act,max_length,for_train,inject_act_list)
                inputs_act.append(zero_act)
                meta_ratio.append(inj_ratio_list)
            else:
                zero_act, _ = self._zero_pad(trace_act,max_length,for_train)
                inputs_act.append(zero_act)
        
        if inject_act_list is not None:
            tensor_set = TensorDataset(torch.cat(inputs_act,dim=0).squeeze(-1),torch.cat(inputs_attr,dim=0),torch.cat(inputs_label,dim=0).squeeze(-1),torch.cat(inputs_length,dim=0).squeeze(-1),torch.cat(meta_ratio,dim=0))
        else:
            tensor_set = TensorDataset(torch.cat(inputs_act,dim=0).squeeze(-1),torch.cat(inputs_attr,dim=0),torch.cat(inputs_label,dim=0).squeeze(-1),torch.cat(inputs_length,dim=0).squeeze(-1))
        return tensor_set
    
    def _sort_tensor_by_length(self, tensor_set,inject_act_list=None):
        if inject_act_list is not None:
            act, attr, label, lengths, ratio = tensor_set.tensors
            sorted_indices = lengths.argsort(descending=True)
            sorted_act = act[sorted_indices]
            sorted_attr = attr[sorted_indices]
            sorted_label = label[sorted_indices]
            sorted_lengths = lengths[sorted_indices]
            sorted_ratio = ratio[sorted_indices]
            return TensorDataset(sorted_act, sorted_attr, sorted_label, sorted_lengths, sorted_ratio)
        else:
            act, attr, label, lengths = tensor_set.tensors
            sorted_indices = lengths.argsort(descending=True)
            sorted_act = act[sorted_indices]
            sorted_attr = attr[sorted_indices]
            sorted_label = label[sorted_indices]
            sorted_lengths = lengths[sorted_indices]
            return TensorDataset(sorted_act, sorted_attr, sorted_label, sorted_lengths)
    
    
    
    def SetGenerator(self):
        df_train,df_test = self._log_reader()
        df_train,df_test = self._tokenization(df_train,df_test,COL.ACT.value)
        df_train,df_test = self._tokenization(df_train,df_test,COL.RES.value)
        df_train,df_test = self._time_feature(df_train),self._time_feature(df_test)
        df_train,df_test,_ = self._scaler(df_train,df_test,COL.TSP.value)
        df_train,df_test,_ = self._scaler(df_train,df_test,COL.TSSC.value)
        if self._task == TASK.NAP.value:
            df_train,df_test,output_dim = self._nap_label(df_train,df_test)
            lab_scaler = None
        if self._task == TASK.OP.value:
            df_train,df_test = self._op_label(df_train,df_test)
            output_dim = 1
            lab_scaler = None            
        elif self._task == TASK.ERTP.value:
            df_train,df_test = self._ertp_label(df_train,df_test)
            output_dim = 1
            df_train,df_test,lab_scaler = self._scaler(df_train,df_test,COL.LABEL.value)
        elif self._task == TASK.CRTP.value:
            df_train,df_test = self._crtp_label(df_train,df_test)
            output_dim = 1        
            df_train,df_test,lab_scaler = self._scaler(df_train,df_test,COL.LABEL.value)
        num_act, num_res, max_length,inject_act_list = self._meta_scrap(df_train,df_test)
        train_tensor = self._prfx_bucket(df_train,num_res,max_length,True)
        test_tensor = self._prfx_bucket(df_test,num_res,max_length,False, inject_act_list)
        
        train_tensor = self._sort_tensor_by_length(train_tensor)
        test_tensor = self._sort_tensor_by_length(test_tensor,inject_act_list)
        train_loader = DataLoader(train_tensor, batch_size= self._batchsize, shuffle=False)
        test_loader = DataLoader(test_tensor, batch_size= self._batchsize, shuffle=False)
        
        num_res+=2 # timefeature dimension increase
        num_act+=1 # zero padding dimension increase
        if self._task == TASK.NAP.value:
            output_dim+=1 # zero padding dimension increase
        
        
        
        meta = {META.OUTDIM.value:output_dim,META.SCALER.value :lab_scaler,
            META.NUMACT.value:num_act, META.ATTRSZ.value:num_res,
            META.MAXLEN.value:max_length }
        return train_loader, test_loader, meta



