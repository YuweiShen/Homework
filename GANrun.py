# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:55:48 2019

@author: Susie
"""

import pandas as pd
import numpy as np
import torch
import argparse
import json
import os
from model import vaetrain
from model import VAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='a json config file',default ='E:/anaconda/param.json')
    args = parser.parse_args()
    with open(args.config) as f:
        param = json.load(f)
    try:
        os.mkdir("expdir")
    except:
        pass
#    for param in config:
    path = "expdir/"+param["name"]+"/"
    try:
        os.mkdir("expdir/"+param["name"])
    except:
        pass
    encoder_dim = param["encoder_dim"]
    encoder_out_dim = param["encoder_out_dim"]
    decoder_dim = param["decoder_dim"]
    latent_dim = param["latent_dim"]
    lr = param["lr"]
    epochs = param["epochs"]
    steps_per_epoch = param["steps_per_epoch"]
    data = pd.read_csv(param["train"], header=None)
    input_dim = param['input_dim']
    cuda = not param["no_cuda"] and torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    fields = []
    feed_data = []
    for i, col in enumerate(list(data)):
        if i in param["continuous_cols"]:
            fields.append(data[i])
            feed_data.append(data[i])
                    #先不编码看一下fields.append((col, NumericalField("gmm", 5))
        else:
#            fields.append(CategoricalField("one-hot",noise=param["noise"]))
            col1 = CategoricalField("one-hot", noise=param["noise"])
            fields.append(col1)
            col1.get_data(data[i])
            col1.learn()
            features = col1.convert(np.asarray(data[i]))
            cols = features.shape[1]
            rows = features.shape[0]
            for j in range(cols):
                feed_data.append(features.T[j])
    feed_data = pd.DataFrame(feed_data).T
    for t in range(5):
        vae = VAE(input_dim,encoder_dim,encoder_out_dim,latent_dim,decoder_dim,device).to(device)
        vaetrain(t, path,  vae, feed_data, fields, epochs, lr, steps_per_epoch = steps_per_epoch,device = device)
