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

sys.path.append('/root/zengzihui/ISST/ISST_Baselines/mTAN/src/get_SWaT')

from get_SWaT import get_swat_data

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('boarder_log/tanenc_class')

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
    parser.add_argument('--learn-emb', default = False, action='store_true')
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--freq', type=float, default=10.)
    parser.add_argument('--dataset', type=str, default='physionet')
    parser.add_argument('--old-split', type=int, default=1)
    parser.add_argument('--nonormalize', action='store_true')
    parser.add_argument('--classify-pertp', action='store_true')
    parser.add_argument('--less', action='store_true')
    parser.add_argument('--beta', default = 1.0, type=float)
    args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random()*100000)
    # print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)    
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')


    # 1. Get Data    
    data_obj = get_swat_data(batch_size = args.batch_size, beta = args.beta, LESS = args.less)
        
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    

    # 2. Set Model
    rec = models.enc_mtan_classif_activity(dim, 4 * dim, args.embed_time, args.num_heads, args.learn_emb, args.freq).to(device)
    print(f'''
            dim {dim}, 
            args.rec_hidden {4 * dim}, 
            args.embed_time {args.embed_time}, 
            args.num_heads {args.num_heads}, 
            args.learn_emb {args.num_heads}, 
            args.freq {args.freq}
                ''')
    print(rec)
    params = (list(rec.parameters()))
    print('parameters:', utils.count_parameters(rec))



    # 3. Set Optimizer and Loss_function
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
    


    total_time = 0.
    epoch  = 0
    for itr in range(1, args.niters + 1):
        epoch = epoch + 1
        loss_train = []
        start_time = time.time()


        # Training Phase
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
        writer.add_scalar('Loss per iteration', np.mean(loss_train), epoch)

        # ## Val Phase
        # loss_val = []
        # label_val = []
        # with torch.no_grad():
        #     for val_batch, label in val_loader:
        #         val_batch, label = val_batch.to(device), label.to(device)
        #         observed_data, observed_mask, observed_tp = val_batch[:, :, :dim], val_batch[:, :, dim:2*dim], val_batch[:, :, -1]
        #         loss = rec(torch.cat((observed_data, observed_mask), 2), observed_tp).cpu().numpy()
                
        #         label_val.append(label.cpu().numpy())
        #         loss_val.append(loss)

        #     loss_val = np.concatenate(loss_val)
        #     label_val = np.concatenate(label_val)

        #     roc_val = roc_auc_score(label_val,loss_val)
        #     writer.add_scalar('AUC-ROC', roc_val, epoch)
        #     print(roc_val)



        ## Testing Phase
        loss_test = []
        label_test = []
        rec.eval()
        with torch.no_grad():
            for test_batch, label in test_loader:
                test_batch, label = test_batch.to(device), label.to(device)
                observed_data, observed_mask, observed_tp = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
                loss = -rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

                loss, _  = loss.max(1)

                loss = loss.cpu().numpy()
                
                # print(loss.shape)

                label_test.append(label.cpu().numpy())
                loss_test.append(loss)

            loss_test = np.concatenate(loss_test)
            label_test = np.concatenate(label_test)

            pd.DataFrame([label_test, loss_test]).T.to_csv(f"label_test_beta_{args.beta}.csv")
            roc_test = roc_auc_score(label_test,loss_test)
            writer.add_scalar('AUC-ROC', roc_test, epoch)
            print(f"Iter {epoch}: Train Loss  = {np.mean(loss_train)}, Test Loss = {np.mean(loss_test)}, roc_score = {roc_test}")
            # print(roc_test)


writer.close()