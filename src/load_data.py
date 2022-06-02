from http import server
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def Sampled_data(values, beta):
    indices = np.where(np.isfinite(values).ravel())[0] 
    to_replace = np.random.permutation(indices)[:int(indices.size * beta)] 
    values[np.unravel_index(to_replace, values.shape)] = -1

    return values

def load_swat(batch_size, beta = 0):
    print("# Loading SWaT DataSet... \n")
    
    train_path = "/root/zengzihui/ISST/GANF/data/SWaT_Dataset_Normal_v1.csv"
    test_path = "/root/zengzihui/ISST/GANF/data/SWaT_Dataset_Attack_v0.csv"

    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)

    data = pd.concat([train_data, test_data])

    # 1. Process time 
    time_index = data.index.to_list()
    time_index = [datetime.strptime(x, "%d/%m/%Y %H:%M:%S %p") for x in time_index]
    time_index = [x - time_index[0] for x in time_index]
    data.index = time_index

    # 2. Process Label
    data = data.rename(columns={"Normal/Attack":"label"})
    data.label = data['label'].map({"Normal": 0,"Attack": 1})


    # 3. Arrange DataLoader
    train_len = len(train_data)
    labels = data.label
    values = data.iloc[:, :-1].values
    if beta > 0:
        values = Sampled_data(values, beta)
        sample_rate = (values[values == -1].size / values.size)
        print(f" - Sampled Data by Rate = {sample_rate} \n")
    else:
        values = values


    # std_scaleer = StandardScaler()
    # values = std_scaleer.fit_transform(values)

    n_sensor = 51

    train_df = pd.DataFrame(values[:int(0.8*train_len)],\
        index = data.index[:int(0.8*train_len)])
    train_label = labels[:int(0.8*train_len)]

    val_df = pd.DataFrame(values[int(0.8*train_len):train_len], \
        index = data.index[int(0.8*train_len):train_len])
    val_label = labels[int(0.8*train_len):train_len]

    test_df = pd.DataFrame(values[train_len:], \
        index = data.index[train_len:])
    test_label = labels[train_len:]
  
    train_loader = DataLoader(WaterLabel(train_df, train_label, timestamp=data.index[:int(0.8*train_len)]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WaterLabel(val_df, val_label, timestamp = data.index[int(0.8*train_len):train_len]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WaterLabel(test_df,test_label, timestamp = data.index[train_len:]), batch_size=batch_size, shuffle=False)
    # return train_loader, val_loader, test_loader, n_sensor
    data_obj = {
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim":train_df.shape[1]
    }

    return data_obj

def load_wadi(batch_size, beta = 0):

    print("# Loading WADI DataSet... \n")
    train_path = "/root/zengzihui/ISST/Baselines/usad/data/wadi/train.csv"
    test_path = "/root/zengzihui/ISST/Baselines/usad/data/wadi/test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    data = pd.concat([train_data, test_data])

    data = data.fillna(0)
    data = data.drop(["Row", "Date", "Time"], axis = 1)

    # 1. Process TimeArr
    time_index = [datetime.fromtimestamp(x) for x in data.index]
    time_index = [x - time_index[0] for x in time_index]
    data.index = time_index

    # 2. Process Label
    data = data.rename(columns={"Attack LABLE (1:No Attack, -1:Attack)": "label"})
    data.label = data.label.map({-1:0, 1:1})

    # 3. Arrange DataLoader
    train_len = len(train_data)
    labels = data.label
    values = data.iloc[:, :-1].values
    if beta > 0:
        values = Sampled_data(values, beta)
        sample_rate = (values[values == -1].size / values.size)
        print(f" - Sampled Data by Rate = {sample_rate} \n")
    else:
        values = values


    std_scaleer = StandardScaler()
    # values = std_scaleer.fit_transform(values)

    n_sensor = 127

    train_df = pd.DataFrame(values[:train_len],\
        index = data.index[:train_len])
    train_label = labels[:train_len]

    val_df = pd.DataFrame(values[train_len:int(1.2*train_len)], \
        index = data.index[train_len:int(1.2*train_len)])
    val_label = labels[train_len:int(1.2*train_len)]

    test_df = pd.DataFrame(values[int(1.2* train_len):], \
        index = data.index[int(1.2* train_len):])
    test_label = labels[int(1.2* train_len):]
  
    train_loader = DataLoader(WaterLabel(train_df, train_label, timestamp=data.index[:train_len]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WaterLabel(val_df, val_label, timestamp = data.index[train_len:int(1.2*train_len)]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WaterLabel(test_df,test_label, timestamp = data.index[int(1.2* train_len):]), batch_size=batch_size, shuffle=False)

    # return train_loader, val_loader, test_loader, n_sensor
    data_obj = {
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim":train_df.shape[1]
    }

    return data_obj

def load_smd(batch_size, beta = 0, server_name = "machine-1-1"):

    # server_name = "machine-1-1"
    print(f"# Loading SMD {server_name} DataSet... \n")
    train_path = f"/root/zengzihui/dataset/SMD/ServerMachineDataset/train/{server_name}.txt"
    test_path = f"/root/zengzihui/dataset/SMD/ServerMachineDataset/test/{server_name}.txt"
    labels_path = f"/root/zengzihui/dataset/SMD/ServerMachineDataset/test_label/{server_name}.txt"

    train = pd.read_csv(train_path, header=None)
    train["labels"] = [0 for _ in range(len(train))]

    test = pd.read_csv(test_path, header=None)
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.rename({0: "labels"}, axis=1)
    test = pd.concat([test, labels], axis=1)

    data = pd.concat([train, test])

    # 1. Process Time
    time_index = [datetime.fromtimestamp(x) for x in data.index]
    time_index = [x - time_index[0] for x in time_index]
    data.index = time_index

    # 2. Arrange DataLoader
    train_len = len(train)
    labels = data.labels
    values = data.iloc[:, :-1].values

    if beta > 0:
        values = Sampled_data(values, beta)
        sample_rate = (values[values == -1].size / values.size)
        print(f" - Sampled Data by Rate = {sample_rate} \n")
    else:
        values = values

    std_scaleer = StandardScaler()
    # values = std_scaleer.fit_transform(values)

    n_sensor = 38

    train_df = pd.DataFrame(values[:train_len],\
        index = data.index[:train_len])
    train_label = labels[:train_len]

    val_df = pd.DataFrame(values[train_len:int(1.2*train_len)], \
        index = data.index[train_len:int(1.2*train_len)])
    val_label = labels[train_len:int(1.2*train_len)]

    test_df = pd.DataFrame(values[int(1.2* train_len):], \
        index = data.index[int(1.2* train_len):])
    test_label = labels[int(1.2* train_len):]

    train_loader = DataLoader(WaterLabel(train_df, train_label, timestamp=data.index[:train_len]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WaterLabel(val_df, val_label, timestamp = data.index[train_len:int(1.2*train_len)]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WaterLabel(test_df,test_label, timestamp = data.index[int(1.2* train_len):]), batch_size=batch_size, shuffle=False)

    # return train_loader, val_loader, test_loader, n_sensor
    data_obj = {
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim":train_df.shape[1]
    }

    return data_obj

class WaterLabel(Dataset):
    def __init__(self, df, label, timestamp, window_size=60, stride_size=10):
        super(WaterLabel, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx = self.preprocess(df,label)
        self.label = label
        # self.label = 1.0-2*self.label 
        self.mask = self.get_mask()
        self.timestamp = [int(x.total_seconds()) for x in timestamp] # bug fixed time
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask]# , label[start_idx[idx_mask]]
    
    def get_mask(self):
        # 2022.4.20 15: 49
        # writen by sync 
        mask = 1 - (self.data == -1).astype(float)
        return mask

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1])
        mask = self.mask[start:end].reshape([self.window_size,-1])
        time = torch.FloatTensor(self.timestamp[start:end]).reshape([self.window_size, -1])
        label = self.label[end-1]
        
        return torch.cat(
            [torch.FloatTensor(data), 
            torch.FloatTensor(mask),
            time], 1), torch.Tensor([label]).long().squeeze()

        # return torch.FloatTensor(data).transpose(0,1), self.label[index]

