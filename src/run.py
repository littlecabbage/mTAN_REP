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
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
from random import SystemRandom
import models
import utils
import sys
sys.path.append('/root/zengzihui/ISST/ISST_Baselines/mTAN/src/load_data')
from load_data import load_smd, load_wadi, load_swat
from sklearn.metrics import f1_score, recall_score, precision_score
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

class Logger(object):
    def __init__(self, file_name = 'temp.log', stream = sys.stdout) -> None:
        self.terminal = stream
        self.log = open(f'log/{file_name}', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def search_best_f1(val_predict, test_label, test_predict):
    """
    It takes in a list of labels and a list of predictions, and returns the best F1 score, precision,
    and recall, 
    ** along with the threshold that produced the best F1 score **
    
    :param label: the actual labels of the data
    :param predict: the predicted probabilities of the positive class
    """

    best_thr = max(val_predict)
    predict_ = [0 if x < best_thr else 1 for x in test_predict]
    f1, pre, rec = f1_score(test_label, predict_), precision_score(test_label, predict_), recall_score(test_label, predict_)
    return best_thr, f1, pre, rec

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

sys.stdout = Logger(str(log_file_name))
sys.stderr = Logger(str(log_file_name))

if __name__ == '__main__':
    experiment_id = int(SystemRandom().random()*100000)
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
    # print(f'''
    #         dim {dim}, 
    #         args.rec_hidden {4 * dim}, 
    #         args.embed_time {args.embed_time}, 
    #         args.num_heads {args.num_heads}, 
    #         args.learn_emb {args.num_heads}, 
    #         args.freq {args.freq}
    #             ''')
    # print(rec)
    
    params = (list(rec.parameters()))
    # print('parameters:', utils.count_parameters(rec))



    # 3. Set Optimizer and Loss_function
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
    
    # 4. Start Training ... ===============================================================================================================
    print("# Start Training ... ===========================================================================================================\n")
    total_time = 0.
    epoch  = 0
    Results = []
    for itr in range(1, args.niters + 1):
        epoch = epoch + 1
        loss_train = []
        start_time = time.time()

        ## 4.1 Training Phase
        rec.train()
        for train_batch, _ in train_loader:
            train_batch = train_batch.to(device)
            observed_data, observed_mask, observed_tp = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            loss = -rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

            loss = loss.mean()
            loss_train.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # tensorboarder visilizion    

        ## 4.2 Val Phase
        loss_val = []
        label_val = []
        rec.eval()
        with torch.no_grad():
            for val_batch, label in val_loader:
                val_batch, label = val_batch.to(device), label.to(device)
                observed_data, observed_mask, observed_tp = val_batch[:, :, :dim], val_batch[:, :, dim:2*dim], val_batch[:, :, -1]
                loss = -rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

                # Use max
                # loss, _  = loss.max(1)
                # loss = loss.cpu().numpy()
                # label_val.append(label.cpu().numpy())
                # loss_val.append(loss)

                # Use mean
                loss = loss.mean(1)
                loss = loss.cpu().numpy()
                label_val.append(label.cpu().numpy())
                loss_val.append(loss)


            loss_val = np.concatenate(loss_val)
            label_val = np.concatenate(label_val)
        writer.add_scalars('Loss', {'Train Loss': np.mean(loss_train), 'Val Loss': np.mean(loss_val)}, epoch)
            
        ## 4.3 Testing Phase
        loss_test = []
        label_test = []
        rec.eval()
        with torch.no_grad():
            for test_batch, label in test_loader:
                test_batch, label = test_batch.to(device), label.to(device)
                observed_data, observed_mask, observed_tp = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
                loss = -rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

                # Use max
                # loss, _  = loss.max(1)
                # loss = loss.cpu().numpy()
                # label_test.append(label.cpu().numpy())
                # loss_test.append(loss)

                # Use min
                loss = loss.mean(1)
                loss = loss.cpu().numpy()
                label_test.append(label.cpu().numpy())
                loss_test.append(loss)

            loss_test = np.concatenate(loss_test)
            label_test = np.concatenate(label_test)


        thr, f1, pre, recall = search_best_f1(test_label = label_test,test_predict = loss_test,val_predict = loss_val)
        Results.append((f1, pre, recall))
        print("Epoch {%d}: loss_train = {%.4f} loss_val = {%.4f} \t[thr = {%.4f} f1 = {%.4f}, pre = {%.4f}, rec = {%.4f}]\n" % (epoch, np.mean(loss_train), np.mean(loss_val), thr, f1, pre, recall)) 

        # sw.add_scalars('loss', {'lossb1':lossb1.item(), 'lossd1':lossd1.item(), 'loss1':loss1.item(), 'lossb2':lossb2.item(), 'lossd2':lossd2.item(), 'loss2':loss2.item()}, global_step=global_step)

        writer.add_scalars('Metrics', {'F1': f1, 'pre': pre, 'recall': recall}, epoch)


Results = sorted(Results, key=lambda x: x[0], reverse=True)
print(f"Global Best Result: F1 = {Results[0][0]}, Pre = {Results[0][1]}, Rec = {Results[0][2]}")
writer.close()