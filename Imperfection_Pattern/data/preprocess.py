import os
import pickle
import random
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
random.seed(42)


from utils.config import LOGTYPE,LOGCOLUMN,TASK,NAME
class SETGENERATOR:
    def __init__(self, dataset, filepath, task, saveloc):
        self._dataset = dataset
        self._filepath = filepath
        self._task = task
        self._saveloc = saveloc
        if not os.path.exists(f"{self._saveloc}{self._dataset[:-4]}/{self._task}"):
            os.makedirs(f"{self._saveloc}{self._dataset[:-4]}/{self._task}")
        self._saveloc = f"{self._saveloc}{self._dataset[:-4]}/{self._task}"
        self._logtype = LOGTYPE(self._dataset)
        self._logcolumn = LOGCOLUMN(self._dataset)
    def _log_reader(self) -> pd.DataFrame:
        df = pd.read_csv(f"{self._filepath}{self._dataset}",
                         usecols=list(self._logtype.keys()),
                         dtype={k: v for k, v in self._logtype.items() if k != self._logcolumn.TIME.value}, 
                         parse_dates=[self._logcolumn.TIME.value])
        return df
    
    def _missing_value(self,df) -> pd.DataFrame:
        for column, dtype in self._logtype.items():
            if dtype == str:
                df[column] = df[column].fillna("Null_value")
            elif dtype == float:
                df[column] = df[column].fillna(df[column].min()) # missing values are filled with the lowest value for each attribute. You have to later modify it. If not, bias may happen during training
            elif ((dtype == bool) or (dtype == 'timestamp')) and (df[column].isnull().any()):
                raise ValueError(f"Missing values found in column '{column}'. Execution stopped.")
        return df
    

    def _nap_label(self,df) -> Tuple[pd.DataFrame,int]:
        try:
            df.drop(self._logcolumn.OUTCOME.value, axis=1,inplace = True)
        except ValueError:
            pass
        df[NAME.LABEL.value] = df.groupby(self._logcolumn.CASE.value)[self._logcolumn.ACT.value].shift(-1)
        #df[NAME.LABEL.value] = df[NAME.LABEL.value].fillna("End of Case")
        unique_label = df[NAME.LABEL.value].dropna().unique()
        label_to_idx = {a: i for i, a in enumerate(unique_label)}
        df[NAME.LABEL.value] = df[NAME.LABEL.value].map(label_to_idx)
        output_dim = df[NAME.LABEL.value].nunique()
        return df, output_dim
    
    
    def _op_label(self,df) -> pd.DataFrame:  # Should be modified. The outcome prediction label generation is soley for BPIC15. Cannot find outcome label in other logs.
        df["Label"] = df[self._logcolumn.OUTCOME.value].astype(int)
        df.drop(self._logcolumn.OUTCOME.value, axis=1,inplace = True)
        return df
    
    def _ertp_label(self,df) -> pd.DataFrame:
        scaler = MinMaxScaler()
        df[NAME.LABEL.value] = df.groupby(self._logcolumn.CASE.value)[self._logcolumn.TIME.value].shift(-1) - df[self._logcolumn.TIME.value]
        df[NAME.LABEL.value] = df[NAME.LABEL.value].dt.total_seconds()
        df[NAME.LABEL.value] = scaler.fit_transform(df[[NAME.LABEL.value]])
        try:
            df.drop(self._logcolumn.OUTCOME.value, axis=1,inplace = True)
        except ValueError:
            pass
        with open(f"{self._saveloc}/scaler.pkl", 'wb') as file:
            pickle.dump(scaler, file)
        return df
    
    def _crtp_label(self,df) -> pd.DataFrame:
        scaler = MinMaxScaler()
        temp = df.groupby(self._logcolumn.CASE.value)[self._logcolumn.TIME.value].transform('max')
        df[NAME.LABEL.value] = (temp- df[self._logcolumn.TIME.value]).dt.total_seconds()
        df[NAME.LABEL.value] = scaler.fit_transform(df[[NAME.LABEL.value]])
        try:
            df.drop(self._logcolumn.OUTCOME.value, axis=1,inplace = True)
        except ValueError:
            pass
        with open(f"{self._saveloc}/scaler.pkl", 'wb') as file:
            pickle.dump(scaler, file)
        return df
    
    def _time_feature(self,df) ->pd.DataFrame:
        scaler = MinMaxScaler()
        df[NAME.TSP.value] = (df[self._logcolumn.TIME.value] - df.groupby(self._logcolumn.CASE.value)[self._logcolumn.TIME.value].shift(1)).dt.total_seconds()
        df[NAME.TSP.value] = df[NAME.TSP.value].fillna(0)
        df[NAME.TSP.value] = scaler.fit_transform(df[[NAME.TSP.value]])
        scaler = MinMaxScaler()
        df[NAME.TSSC.value] = (df[self._logcolumn.TIME.value] - df.groupby(self._logcolumn.CASE.value)[self._logcolumn.TIME.value].transform('min')).dt.total_seconds()
        df[NAME.TSSC.value] = scaler.fit_transform(df[[NAME.TSSC.value]])
        return df
        
    def _prfx_bucket(self,df) -> Tuple[pd.DataFrame, int]:
        transformed_data = []
        max_length = df.groupby(self._logcolumn.CASE.value).size().max() -1
        for case_id in df[self._logcolumn.CASE.value].unique():
            case_data = df[df[self._logcolumn.CASE.value] == case_id]
            for i in range(1, case_data.shape[0]): # max prefix = last event -1 in the case ; if the trace has 4 events, 3 events will be max-prefix bucket. Why like this? because of event/case remaining time
                prefix_data = case_data.iloc[:i]    # if you want all events, modify as :  range(1, case_data.shape[0]+1), 
                if i < max_length:
                    padding_rows = max_length - i
                   
                    pad_dict = {key: case_id if key == self._logcolumn.CASE.value else
                                "Zero Pad" if key == self._logcolumn.ACT.value else
                                prefix_data[NAME.LABEL.value].iloc[-1] if key == NAME.LABEL.value else
                                np.nan for key in df.columns}
                    
                    padded_rows = pd.DataFrame(pad_dict, index=[0]*padding_rows)
                    prefix_data = pd.concat([prefix_data, padded_rows], ignore_index=True)
                prefix_data[NAME.LABEL.value] = prefix_data[NAME.LABEL.value].iloc[-1]
                if prefix_data[NAME.LABEL.value].nunique()>1:
                    raise ValueError("Label is not unified in the trace")
                transformed_data.append(prefix_data)
        df = pd.concat(transformed_data, ignore_index=True)
        df[NAME.TSP.value] = df[NAME.TSP.value].fillna(0)  # Time feature zero padding
        df[NAME.TSSC.value] = df[NAME.TSSC.value].fillna(0)
        return df, max_length
    
    def _activity_mapping(self,df) -> pd.DataFrame:   
        unique_act = df[self._logcolumn.ACT.value].unique()
        act_to_idx = {a: i for i, a in enumerate(unique_act)}
        df[self._logcolumn.ACT.value] = df[self._logcolumn.ACT.value].map(act_to_idx)  #  def _train_test_split():
        return df
    
    def _onehot_minmax(self, df, separator = ',') -> Tuple[pd.DataFrame, np.ndarray]: # for some columns, there are multiple objects in their items. In min max, data leakage happened (the max value is searched in both train and test set)
        encoded_arrays = []                                                          # Since there are multiple objects in some items, we cannot use boolean type to save memory. (due to summation, int type is necessary)
        for column in self._logcolumn.ATTRLIST.value:                                   # That's why memory explosion happens
            if self._logtype[column] == str:                                               #Hence, attribute& time feature ( Except case, activity, label) will be stored as np.array
                temp_df = df[column].str.split(separator, expand=True).stack()
                temp_df = temp_df.reset_index(level=1, drop=True).str.strip()
                temp_encoded = pd.get_dummies(temp_df, prefix=column)
                temp_encoded = temp_encoded.groupby(temp_encoded.index).sum()
                temp_encoded = temp_encoded.reindex(df.index, fill_value=0)
                encoded_arrays.append(temp_encoded.values)
                #df = df.drop(columns=[column]).join(temp_encoded)  # due to memory usage explosion, attribute, time feature ( Except case, activity, label) will be stored as np.array
            elif self._logtype[column] == float:
                df[column] = df[column].fillna(df[column].min()) # zero padded float attributes are filled with the lowest value for each attribute. You have to later modify it
                scaler = MinMaxScaler()
                df[column] = scaler.fit_transform(df[[column]])
                encoded_arrays.append(np.array(df[column]).reshape(len(df[column]), 1))
                
        encoded_arrays.append(df[[NAME.TSP.value,NAME.TSSC.value]].values)
        attr_encoded_array = np.hstack(encoded_arrays)
        df.drop(self._logcolumn.ATTRLIST.value, axis=1,inplace=True)
        df.drop([self._logcolumn.TIME.value,NAME.TSP.value,NAME.TSSC.value], axis=1,inplace=True)
        
        return df, attr_encoded_array
    
    def _traintest_split_save(self, df, attr_array, max_length, train_ratio=0.8) -> None : # ratio of case : 80% case will be in train set, 20% will be in test set
        split = list(df[self._logcolumn.CASE.value].unique())
        split = split[:int(len(split)*train_ratio)]
        df_train = df[df[self._logcolumn.CASE.value].isin(split)]
        df_test = df[~(df[self._logcolumn.CASE.value].isin(split))]
        
        attr_train = attr_array[df_train.index.tolist()]
        attr_train = attr_train.reshape(-1,max_length,attr_train.shape[1])

        attr_test = attr_array[df_test.index.tolist()]
        attr_test = attr_test.reshape(-1,max_length,attr_test.shape[1])
        
        label_train = df_train[NAME.LABEL.value].values.reshape(-1,max_length,1)
        label_test = df_test[NAME.LABEL.value].values.reshape(-1,max_length,1)

        act_train = df_train[self._logcolumn.ACT.value].values.reshape(-1,max_length,1)
        act_test = df_test[self._logcolumn.ACT.value].values.reshape(-1,max_length,1)
                                                                                                            # The reason for saving train/test set separately with the type of array?
                                                                                                            # activity tensor will pass embedding layer and concatenate with attribute tensor
        np.savez(f"{self._saveloc}/{self._dataset[:-4]}_train.npz",
                 **{self._logcolumn.ACT.value: act_train},
                 **{NAME.LABEL.value: label_train},
                 **{NAME.ATTR.value: attr_train})
        np.savez(f"{self._saveloc}/{self._dataset[:-4]}_test.npz",
                 **{self._logcolumn.ACT.value: act_test},
                 **{NAME.LABEL.value: label_test},
                 **{NAME.ATTR.value: attr_test})

        return None
    
    def _meta_extractor(self,output_dim, num_act, max_length, attr_size) -> None:
        meta_dict = {NAME.OUTDIM.value:output_dim,
                     NAME.NUMACT.value:num_act,
                     NAME.MAXLEN.value:int(max_length),
                     NAME.ATTRSZ.value:attr_size}
        with open(f'{self._saveloc}/meta_info.json', 'w') as json_file:
            json.dump(meta_dict, json_file, indent=4)
        return None
    
    def SetGenerator(self):
        df = self._log_reader()
        df = self._missing_value(df)
        if self._task == TASK.NAP.value:
            df, output_dim = self._nap_label(df)
        elif self._task == TASK.OP.value:
            df = self._op_label(df)
            output_dim = 1
        elif self._task == TASK.ERTP.value:
            df = self._ertp_label(df)
            output_dim = 1
        elif self._task == TASK.CRTP.value:
            df = self._crtp_label(df)
            output_dim = 1
            
        df = self._time_feature(df)
        df, max_length = self._prfx_bucket(df)
        df = self._activity_mapping(df)
        df, attr_array = self._onehot_minmax(df)
        self._traintest_split_save(df,attr_array,max_length)
        
        num_act = df[self._logcolumn.ACT.value].nunique()
        attr_size = attr_array.shape[1]

        self._meta_extractor(output_dim, num_act, max_length, attr_size)
        


