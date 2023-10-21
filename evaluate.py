from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedMSETest
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np

def horiz_eval(loss_total, n_horiz):
    loss_total = loss_total.cpu().numpy()
    avg_res = np.zeros(n_horiz)
    n_all = loss_total.shape[0]
    n_frames = n_all//n_horiz
    for i in range(n_horiz):
        if i == 0:
            st_id = 0
        else:
            st_id = n_frames*i

        if i == n_horiz-1:
            en_id = n_all-1
        else:
            en_id = n_frames*i + n_frames - 1

        avg_res[i] = np.mean(loss_total[st_id:en_id+1])

    return avg_res

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)


args['input_embedding_size'] = 32

args['train_flag'] = False


# Evaluation metric:

metric = 'rmse'

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/lr_atcn-sta_lstm_epoch_11.tar'))
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('/mnt/AI_2TB/dataset/traj/TestSet.mat', t_f=50)
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn) # 

lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()

vehid = []
pred_x = []
pred_y = []
T = []
dsID = []
ts_cen = []
ts_nbr = []
wt_ha = []


for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, veh_id, t, ds = data

    if not isinstance(hist, list): # nbrs are not zeros
        vehid.append(veh_id) # current vehicle to predict

        T.append(t) # current time
        dsID.append(ds)
    

    # Initialize Variables
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()



        fut_pred, weight_ts_center, weight_ts_nbr, weight_ha= net(hist, nbrs, mask, lat_enc, lon_enc)
        l, c = maskedMSETest(fut_pred, fut, op_mask)

        fut_pred_x = fut_pred[:,:,0].detach() ## Returns ew tensor detached from current graph
        fut_pred_x = fut_pred_x.cpu().numpy() ## Create copy in cpu, make it a numpy array

        fut_pred_y = fut_pred[:,:,1].detach()
        fut_pred_y = fut_pred_y.cpu().numpy()
        pred_x.append(fut_pred_x)
        pred_y.append(fut_pred_y)


        ts_cen.append(weight_ts_center[:, :, 0].detach().cpu().numpy())
        ts_nbr.append(weight_ts_nbr[:, :, 0].detach().cpu().numpy())
        wt_ha.append(weight_ha[:, :, 0].detach().cpu().numpy())

        print('Progress: {:3.2f}%'.format(
            ((i+1)*100) / len(tsDataloader)), end='\r')


        lossVals +=l.detach() # revised by Lei
        counts += c.detach()


print ('lossVal is:', lossVals)

pred_rmse = torch.pow((lossVals / counts), 0.5)*0.3048
print(pred_rmse)   # Calculate RMSE and convert from feet to meters
pred_rmse_horiz = horiz_eval(pred_rmse, 5)
print(pred_rmse_horiz)

pred_fde = torch.pow((lossVals ), 0.5)*0.3048/ counts
print(pred_fde)   # Calculate RMSE and convert from feet to meters
pred_fde_horiz = horiz_eval(pred_fde, 5)
print(pred_fde_horiz)



