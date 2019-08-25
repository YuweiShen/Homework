# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 02:45:29 2019

@author: Susie
"""

import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np



class VAE(nn.Module):
    def __init__(self,input_dim,encoder_dim,encoder_out_dim,latent_dim,decoder_dim,device):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoder_dim),
                nn.BatchNorm1d(encoder_dim),
                nn.ReLU(inplace=True), 
                
                nn.Linear(encoder_dim, encoder_out_dim),
#                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True)
                )
        self.fcmu = nn.Linear(encoder_out_dim, latent_dim) #mean
        self.fcsigma = nn.Linear(encoder_out_dim, latent_dim) #logsigma
        self.decoder = nn.Sequential(              
                nn.Linear(latent_dim, encoder_out_dim),
                nn.BatchNorm1d(encoder_out_dim),
                nn.ReLU(inplace = True),
                
                nn.Linear(encoder_out_dim,decoder_dim),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(inplace = True),
                nn.Linear(decoder_dim, input_dim)
                )
        self.device = device
        

    def reparameterize(self,mu,logvar):
        if self.device.type == 'cpu':
            epsi = Variable(torch.randn(mu.size(0), mu.size(1)))
        elif self.device.type == 'gpu':
            epsi = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        z = mu + epsi*torch.exp(logvar/2)
        return z
    
    def forward(self,x):
        out1, out2 =self.encoder(x), self.encoder(x) # data_size,encoder_dim,12
        mu = self.fcmu(out1) #data_size, latent_num
        logvar = self.fcsigma(out2) #data_size, latent_num
        z = self.reparameterize(mu, logvar)#data_size, latent_num
        return self.decoder(z), mu, logvar

def loss_func(reconstruct, x, mu, logvar):
    batch_size = x.size(0)
    BCE = fun.binary_cross_entropy_with_logits(reconstruct, x, reduction='sum') / batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/ batch_size
    return BCE,KLD


def vaetrain(t, path, vae, data, fields, epochs, lr, steps_per_epoch, device):
    # t: times of training
	# path: output file path
	# VAE :net
    # data: train data
    # fields: help decode categorical data
	# epochs
	# lr: learning rate
	# steps_per_epoch
    if device == torch.device('cuda'):
        vae.cuda()
    optimizer = optim.Adam(vae.parameters(),lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)

    for epoch in range(int(epochs)):
        vae.train()
        print("----------pretrain Epoch %d:----------\n"%epoch)
        log = open(path+"train_log_vae.txt","a+")
        log.write("----------pretrain Epoch %d:----------\n"%epoch)
        log.close()
        it = 0
        sample_num, col_num = data.shape[0], data.shape[1]
        while it < steps_per_epoch:
            dtest = data.values
            dtest = torch.FloatTensor(dtest)
            #x = torch.unsqueeze(dtest, dim=1)
            x = dtest
            if device == torch.device('cuda'):
                x = Variable(x).cuda()
            else:
                x = Variable(x)
            optimizer.zero_grad()
            x_, mu, logvar = vae.forward(x)
            recon_loss, kl_loss = loss_func(x_, x, mu, logvar)
            elbo_loss = recon_loss + kl_loss
            mse_loss = fun.mse_loss(torch.sigmoid(x_).detach(),x.detach(), size_average=True)
            elbo_loss.backward()
            optimizer.step()
                
            if it%100 == 0:
                train_text = "VAE iteration {} \t ELBO Loss {elbo_loss:.4f} \t MSE Reconstruct Loss {mse_loss:.4f}\t BCE Reconstruct Loss {reconstruct_loss:.4f}\t KL Loss {kl_loss:.4f}".format(
                        it,
                        elbo_loss=elbo_loss.item(),
                        mse_loss=mse_loss.item(),
                        reconstruct_loss=recon_loss.item(),
                        kl_loss=kl_loss.item())
                print(train_text)
                log = open(path+"train_log_vae.txt","a+")
                log.write(train_text)
                log.close()  
                sample = vae.forward(torch.FloatTensor(torch.randn(sample_num, col_num)))[0]
                sample_data = []
                for i in len(fields):
                    current_ind = 0
                    if type(fields[i]) == field.CategoricalField:
                        dim = fields[i].dim()
                        data_totranse =np.round( torch.sigmoid(sample.loc[:,current_ind:(current_ind+dim)]).detach().numpy())
                        sample_data.append(fields[i].reverse(data_totranse))
                        current_ind += dim
                    else:
                        sample_data.append(sample[current_ind])
                        current_ind += 1
                sample_data = pd.DataFrame(sample_data)
                sample_data.to_csv(path+'sample_data_vae_{}_{}.csv'.format(t,epoch), index = None)
            it += 1
            if it >= steps_per_epoch:
                break
    