#pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from random import SystemRandom
import models
import utils
import datetime

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/root/zengzihui/ISST/ISST_Baselines/mTAN/src/get_SWaT')
from get_SWaT import get_swat_data


from sklearn.metrics import roc_auc_score

## ----------------- log --------------------------
import sys

class Logger(object):
    def __init__(self, file_name = 'temp.log', stream = sys.stdout) -> None:
        self.terminal = stream
        self.log = open(f'text_log/{filename}', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# open log handler before print
filename = str(datetime.datetime.utcnow() + datetime.timedelta(hours=8)).split(".")[0]
log_file_name = f"{filename}.log"
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)
# ----------------- log --------------------------



args = True
if args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rec-hidden', type=int, default=32)
    parser.add_argument('--embed-time', type=int, default=128)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--enc', type=str, default='mtan_enc')
    parser.add_argument('--fname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--n', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--quantization', type=float, default=0.1, 
                        help="Quantization on the physionet dataset.")
    parser.add_argument('--classif', action='store_true', 
                        help="Include binary classification loss")
    parser.add_argument('--learn-emb', action='store_true')
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--freq', type=float, default=10.)
    parser.add_argument('--dataset', type=str, default='physionet')
    parser.add_argument('--old-split', type=int, default=1)
    parser.add_argument('--nonormalize', action='store_true')
    parser.add_argument('--classify-pertp', action='store_true')
    parser.add_argument('--nolog', default=False)
    args = parser.parse_args()

if not args.nolog:
    # open log handler before print
    print("=> Recording tanenc_classification.py....")
    filename = str(datetime.datetime.now())
    log_file_name = f"{filename}.log"
    sys.stdout = Logger(str(log_file_name))
    sys.stderr = Logger(str(log_file_name))
    # ----------------- log --------------------------


# add tensorboard writer
writer = SummaryWriter('boarder_log/tanenc_class')

if __name__ == '__main__':
    experiment_id = int(SystemRandom().random()*100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
        
    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args)
    elif args.dataset == 'activity':
        data_obj = utils.get_activity_data(args, 'cpu')
    elif args.dataset == 'swat':
        data_obj = get_swat_data(batch_size = 128)
        
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    
    # model
    if args.enc == 'mtan_enc':
        rec = models.enc_mtan_classif(
            dim, torch.linspace(0, 1., 128), args.rec_hidden, 
            args.embed_time, args.num_heads, args.learn_emb, args.freq).to(device)
        
    elif args.enc == 'mtan_enc_activity':
        rec = models.enc_mtan_classif_activity(
            dim, args.rec_hidden, args.embed_time, 
            args.num_heads, args.learn_emb, args.freq).to(device)
        

    params = (list(rec.parameters()))
    print('parameters:', utils.count_parameters(rec))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
    
    best_val_loss = float('inf')
    total_time = 0.
    loss_train = []
    rec.train()
    for itr in range(1, args.niters + 1):
        loss_tain = []
        start_time = time.time()

        for train_batch, label in train_loader:
            train_batch = train_batch.to(device)
            # label = label.to(device)
            
            batch_len  = train_batch.shape[0]
            observed_data, observed_mask, observed_tp \
                = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            # out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

            loss = -rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            
            loss.backward(loss.clone().detach())
            optimizer.step()
            # loss_train.append(loss.clone().detach().item())

        rec.eval()
        loss_val = []
        with torch.no_grad():
            for val_batch, label in val_loader:
                val_batch = val_batch.to(device)
                observed_data, observed_mask, observed_tp \
                = val_batch[:, :, :dim], val_batch[:, :, dim:2*dim], val_batch[:, :, -1]
            # out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

            loss = -rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

            loss_val.append(loss)

        loss_val = np.concatenate(loss_val)


        loss_test = []
        with torch.no_grad():
            for test_batch, label in test_loader:
                test_batch = test_batch.to(device)
                observed_data, observed_mask, observed_tp \
                = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
            # out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

            loss = -rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

            loss_test.append(loss)

        loss_test = np.concatenate(loss_test)



        loss_val = np.nan_to_num(loss_val)
        loss_test = np.nan_to_num(loss_test)
        roc_val = roc_auc_score(np.asarray(val_loader.dataset.label.values,dtype=int),loss_val)
        roc_test = roc_auc_score(np.asarray(test_loader.dataset.label.values,dtype=int),loss_test)
        print('Epoch: {}, train -log_prob: {:.2f}, test -log_prob: {:.2f}, roc_val: {:.4f}, roc_test: {:.4f}'\
                .format(iter, np.mean(loss_train), np.mean(loss_val), roc_val, roc_test))




