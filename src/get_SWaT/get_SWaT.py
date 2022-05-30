
from cgi import test
from operator import index
from pkgutil import get_data
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn

from datetime import datetime, timedelta

# > The Water class takes in a dataframe and a label, and returns a dataset object that can be used to
# train a model
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


def get_swat_data(batch_size, beta = 1, LESS = False):
    """
    > The function takes in a batch size and returns a dictionary of dataloaders for train, test, and
    validation
    
    :param batch_size: The number of samples per gradient update
    """

    if LESS:
        less = 0.2
        print(f"> less = {less} Using 0.2 Less DATA !\n")
    else:
        less = 1.0
        print(f"> less = {less} Using 1.0 FULL DATA !\n")

    if beta == 1:
        train_path = "/root/zengzihui/ISST/ISST_Baselines/mTAN/src/get_SWaT/raw_dataset/train.csv"
        test_path = "/root/zengzihui/ISST/ISST_Baselines/mTAN/src/get_SWaT/raw_dataset/test.csv"
        print(f"Beta = {beta}", f"Train Path: {train_path}", f"Test Path : {test_path}", sep='\n')
    elif beta == 0.3:
        train_path = "/root/zengzihui/ISST/ISST_Baselines/mTAN/src/get_SWaT/raw_dataset/train_3_4.csv"
        test_path = "/root/zengzihui/ISST/ISST_Baselines/mTAN/src/get_SWaT/raw_dataset/test_3_4.csv"
        print(f"Beta = {beta}", f"Train Path: {train_path}", f"Test Path : {test_path}", sep='\n')
    elif beta == 0.5:
        train_path = "/root/zengzihui/ISST/ISST_Baselines/mTAN/src/get_SWaT/raw_dataset/train_5_4.csv"
        test_path = "/root/zengzihui/ISST/ISST_Baselines/mTAN/src/get_SWaT/raw_dataset/test_5_4.csv"
        print(f"Beta = {beta}", f"Train Path: {train_path}", f"Test Path : {test_path}", sep='\n')



    train_data = pd.read_csv(train_path, index_col = 0)
    test_data = pd.read_csv(test_path, index_col = 0)

    train_data = train_data.head(int(less * len(train_data)))
    test_data = test_data.head(int(less * len(test_data)))

    data = pd.concat([train_data, test_data])
    # data = test_data

        
    # 1. Process time 
    time_index = data.index.to_list()
    time_index = [datetime.strptime(x, "%d/%m/%Y %H:%M:%S %p") for x in time_index]
    time_index = [x - time_index[0] for x in time_index]
    data.index = time_index

    # 2. Process Label
    data = data.rename(columns={"Normal/Attack":"label"})
    data.label = data['label'].map({"Normal": 0,"Attack": 1})


    # 3. Normalization
    #%%
    feature = data.iloc[:,:51]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)

    train_df = norm_feature.iloc[:int(0.8 * len(train_data))]
    train_label = data.label.iloc[:int(0.8 * len(train_data))]
    train_time = data.index[:int(0.8 * len(train_data))]

    val_df = norm_feature.iloc[int(0.8 * len(train_data)):int(len(train_data))]
    val_label = data.label.iloc[int(0.8 * len(train_data)):int(len(train_data))]
    val_time = data.index[int(0.8 * len(train_data)):int(len(train_data))]
    
    test_df = norm_feature.iloc[int(len(train_data)):]
    test_label = data.label.iloc[int(len(train_data)):]
    test_time = data.index[int(len(train_data)):]
    
    # Init DataLodaer
    train_loader = DataLoader(
        WaterLabel(train_df,train_label, timestamp = train_time), 
        batch_size=batch_size, 
        shuffle=True)
    
    val_loader = DataLoader(
        WaterLabel(val_df,val_label, timestamp= val_time), 
        batch_size=batch_size, 
        shuffle=False)

    test_loader = DataLoader(
        WaterLabel(test_df,test_label, timestamp = test_time), 
        batch_size=batch_size, 
        shuffle=False)


    data_obj = {
        "train_dataloader": train_loader,
        "test_dataloader": test_loader,
        "val_dataloader": val_loader,
        "input_dim":train_df.shape[1]
    }

    return data_obj


if __name__ == "__main__":
    pass
    # For test

    # data_obj = get_swat_data(batch_size=128)
            
    # train_loader = data_obj["train_dataloader"]
    # test_loader = data_obj["test_dataloader"]
    # val_loader = data_obj["val_dataloader"]
    # dim = data_obj["input_dim"]

    # num = 0
    # attack_sum = 0
    # for train_batch, label in train_loader:
    #     # train_batch, label = train_batch.to(device), label.to(device)
    #     # batch_len  = train_batch.shape[0]
    #     # observed_data, observed_mask, observed_tp \
    #     #     = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
        
    #     # print(f'''
    #     #     observed_data: {observed_data.shape},
    #     #     observed_mask:{observed_mask.shape},
    #     #     observed_tp: {observed_tp.shape}
    #     # ''')

    #     # out.shape torch.Size([128, 2])
    #     # label.shape torch.Size([128])

    #     # criterion = nn.CrossEntropyLoss()

    #     # out = torch.zeros([128, 2])
    #     # print(f'''
    #     #     out.shape: {out.shape}
    #     #     label.shape: {label.shape}
    #     # ''')
    #     # print(criterion(out, label))
    #     num = num + 1
    #     Attack = len(label[label == 1])
    #     Normal = len(label[label == 0])
    #     if Attack > 0:
    #         attack_sum += Attack
    #         print(f"{num}: Attack {len(label[label == 1])}, Normal {len(label[label == 0])}")

    # print(attack_sum)
    # print(test_loader.dataset.label)
    # get_data



