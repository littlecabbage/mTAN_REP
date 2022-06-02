## Faster Train(using 0.2 * raw_data)
# python3 run_water.py --niters 1000 --lr 0.001 --batch-size 256   --num-heads 1 --dataset swat --seed 0 --less

## FULl data Beta = 1.0
# python3 run_water.py --niters 1000 --lr 0.001 --batch-size 256   --num-heads 1 --dataset swat --seed 0 


## Sample data data Beta = 0.3
# python3 run_water.py --niters 1000 --lr 0.001 --batch-size 256   --num-heads 1 --dataset swat --seed 0 --beta 0.3


## Sample data data Beta = 0.5
# python3 run_water.py --niters 1000 --lr 0.001 --batch-size 256   --num-heads 1 --dataset swat --seed 0 --beta 0.5

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
from random import SystemRandom
import models
import utils
import sys
sys.path.append('/root/zengzihui/ISST/ISST_Baselines/mTAN/src/load_data')
from load_data import load_smd, load_wadi, load_swat
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from datetime import datetime

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('boarder_log/tanenc_class')

args = True
if args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rec-hidden', type=int, default=32)
    parser.add_argument('--embed-time', type=int, default=128)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--enc', type=str, default='mtan_enc')
    parser.add_argument('--fname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--n', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--quantization', type=float, default=0.1, 
                        help="Quantization on the physionet dataset.")
    parser.add_argument('--classif', action='store_true', 
                        help="Include binary classification loss")
    parser.add_argument('--learn-emb', default = False, action='store_true')
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--freq', type=float, default=10.)
    # parser.add_argument('--dataset', type=str, default='physionet')
    parser.add_argument('--old-split', type=int, default=1)
    parser.add_argument('--nonormalize', action='store_true')
    parser.add_argument('--classify-pertp', action='store_true')

    # parser.add_argument('--less', action='store_true')
    parser.add_argument('--beta', default = 0.0, type=float) 
    parser.add_argument('--dataset', default = "swat", type=str) 
    parser.add_argument('--server_name', default = "machine-1-1", type=str) 
args = parser.parse_args()

def loss_func(y_pred, y_true):
    # loss = torch.nn.BCELoss(y_pred, y_true)
    L = torch.nn.MSELoss()
    loss = L(y_pred, y_true)

    return loss


def train(model, train_loader, loss_train, optimizer, dim):
    model.train()
    for train_batch, label in train_loader:
        train_batch = train_batch.to(device)
        observed_data, observed_mask, observed_tp = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
        loss = -rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
        label = label.unsqueeze(-1)

        # loss = loss.mean()
        
        loss = torch.mean(loss, dim=1, keepdim=True)
        loss = loss_func(loss.float().to(device), label.float().to(device))
        # print(loss)

        loss_train.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss_train

def val(model, val_loader, loss_val, label_val, writer):
    with torch.no_grad():
        model.eval()
        for val_batch, label in val_loader:
            val_batch, label = val_batch.to(device), label.to(device)
            observed_data, observed_mask, observed_tp = val_batch[:, :, :dim], val_batch[:, :, dim:2*dim], val_batch[:, :, -1]
            loss = -model(torch.cat((observed_data, observed_mask), 2), observed_tp)

            # Use mean
            # loss = loss.mean(1)
            # loss = loss.cpu().numpy()

            loss = torch.mean(loss, dim=1, keepdim=True)
            # loss = loss_func(loss.float().to(device), label.float().to(device))
            # print(loss, loss.shape)
            
            
            label_val.append(label.cpu().numpy())
            loss_val.append(loss.cpu().numpy())

        loss_val = np.concatenate(loss_val)
        label_val = np.concatenate(label_val)
        
    
    return model, loss_val, label_val

def test(model, save_path, thresh_hold):
    check_point = torch.load(save_path)
    model.load_state_dict(check_point)
    loss_test = []
    label_test = []

    with torch.no_grad():
        for test_batch, label in test_loader:
            test_batch, label = test_batch.to(device), label.to(device)
            observed_data, observed_mask, observed_tp = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
            loss = -model(torch.cat((observed_data, observed_mask), 2), observed_tp)

            # Use max
            # loss, _  = loss.max(1)
            # loss = loss.cpu().numpy()
            # label_test.append(label.cpu().numpy())
            # loss_test.append(loss)

            # Use min
            # loss = loss.mean(1)
            # loss = loss.cpu().numpy()

            loss = torch.mean(loss, dim=1, keepdim=True)
            # loss = loss_func(loss.float().to(device), label.float().to(device))
            
            label_test.append(label.cpu().numpy())
            loss_test.append(loss.cpu().numpy())

        loss_test = np.concatenate(loss_test)
        label_test = np.concatenate(label_test)

    pd.DataFrame(loss_test).to_csv(f"{log_file_name.split('.')[0]}_loss_test.csv")
    pd.DataFrame(label_test).to_csv(f"{log_file_name.split('.')[0]}_label_test.csv")

    predict_ = [0 if x < thresh_hold else 1 for x in loss_test]
    f1, pre, recall = f1_score(label_test, predict_), precision_score(label_test, predict_), recall_score(label_test, predict_)
    # add matrxi analyze
    print(f"Test f1 = {f1}, pre = {pre}, rec = {recall}\n")
    print(classification_report(label_test, predict_))


if args.beta > 0:
    tm = datetime.now().strftime("%m-%dT%H:%M:%S")
    if args.dataset == "swat":
        log_file_name = f"{args.dataset}_beta_{int(args.beta*10)}@{tm}.log"
    elif args.dataset == "wadi":
        log_file_name = f"{args.dataset}_beta_{int(args.beta*10)}@{tm}.log"
    else:
        log_file_name = f"{args.server_name}_beta_{int(args.beta*10)}@{tm}.log"
else:
    tm = datetime.now().strftime("%m-%dT%H:%M:%S")
    if args.dataset == "swat":
        log_file_name = f"{args.dataset}@{tm}.log"
    elif args.dataset == "wadi":
        log_file_name = f"{args.dataset}@{tm}.log"
    else:
        log_file_name = f"{args.server_name}@{tm}.log"

class Logger(object):
    def __init__(self, file_name = 'temp.log', stream = sys.stdout) -> None:
        self.terminal = stream
        self.log = open(f'log/{file_name}', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(str(log_file_name))
sys.stderr = Logger(str(log_file_name))

if __name__ == '__main__':
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')


    # 1. Get Data    
    # data_obj = get_swat_data(batch_size = args.batch_size, beta = args.beta, LESS = args.less)
    if args.dataset == "swat":
        data_obj = load_swat(batch_size = args.batch_size, beta = args.beta)
    elif args.dataset == "wadi":
        data_obj = load_wadi(batch_size = args.batch_size, beta = args.beta)
    elif args.dataset == "smd":
        data_obj = load_smd(batch_size = args.batch_size, beta = args.beta, server_name = args.server_name)
        
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    

    # 2. Set Model
    rec = models.enc_mtan_classif_activity(dim, 4 * dim, args.embed_time, args.num_heads, args.learn_emb, args.freq).to(device)
    
    # 3. Set Optimizer and Loss_function
    params = (list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Start Training ... 
    print("# Start Training ... \n")
    
    ## 4.0 set temp params
    epoch  = 0
    
    # Performance Metrics
    min_loss = float("inf")
    save_path = f"best_model/{log_file_name.split('.')[0]}.pt"
    thresh_hold = -1

    # Time Metrics
    used_time = []

    for itr in range(1, args.niters + 1):
        epoch = epoch + 1
        
        ## 4.1 Training Phase
        start = time.time() # record start training time
        loss_train = []
        rec, loss_train = train(rec, train_loader, loss_train, optimizer, dim)
        end = time.time() # record edn training time
        used_time.append(end-start)

        ## 4.2 Val Phase
        loss_val, label_val = [], []
        rec, loss_val, _ = val(rec, val_loader, loss_val, label_val, writer)

        ### 4.2.1 Visulization Performance of the Model
        writer.add_scalars('Loss', {'Train Loss': np.mean(loss_train), 'Val Loss': np.mean(loss_val)}, epoch)
        
        #### 4.2.2 Save the best_model
        if np.mean(loss_val) < min_loss:
            print(" > Epoch(%d/%d): loss_train={%.4f} loss_val={%.4f}  (Save Model at Epoch(%d): min_loss=>val_loss: {%.4f}=>{%.4f})  %.2fs\n" \
                % (epoch, args.niters, np.mean(loss_train), np.mean(loss_val), epoch, min_loss, np.mean(loss_val), end-start)) 
            
            min_loss = np.mean(loss_val)
            torch.save(rec.state_dict(), save_path)

            thresh_hold = max(loss_val) # update thresh_hold

            pd.DataFrame(loss_train).to_csv(f"{log_file_name.split('.')[0]}_loss_train.csv")
            pd.DataFrame(loss_val).to_csv(f"{log_file_name.split('.')[0]}_loss_val.csv")
            pd.DataFrame([thresh_hold]).to_csv(f"{log_file_name.split('.')[0]}_thresh_old.csv")
        else:
            print(" > Epoch(%d/%d): loss_train={%.4f} loss_val={%.4f}  %.2fs\n" % (epoch, args.niters, np.mean(loss_train), np.mean(loss_val), end-start)) 

    ## 4.3 Testing Phase
    test(rec, save_path, thresh_hold)
    print("Total Time %.2fs for %d epoches, Avg Time %.2fs per epoch" \
        % (np.sum(used_time), args.niters, np.mean(used_time))) 