#!/usr/bin/env python
# coding: utf-8

import yaml 
import time
import copy
import joblib
import numpy as np
import pandas as pd
import ase
from ase import Atoms, io, build

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# structrepgen
from structrepgen.utils.dotdict import dotdict
from structrepgen.utils.utils import torch_device_select
from structrepgen.reconstruction.reconstruction import Reconstruction
from structrepgen.descriptors.ensemble import EnsembleDescriptors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from mpl_toolkits import mplot3d

torch.set_printoptions(precision=4, sci_mode=False)


class Encoder1(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        dims_arr = np.linspace(hidden_dim, latent_dim, hidden_layers+1, dtype=int)[0:-1]
        dims_arr = np.concatenate(([input_dim + y_dim], dims_arr))
        
        for i, (in_size, out_size) in enumerate(zip(dims_arr[:-1], dims_arr[1:])):
            self.hidden.add_module(
                name='Linear_'+str(i),
                module=nn.Linear(in_size, out_size)
            )
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.mu = nn.Linear(dims_arr[-1], latent_dim)
        self.var = nn.Linear(dims_arr[-1], latent_dim)

    def forward(self, x):
        x = self.hidden(x)
        mean = self.mu(x)
        log_var = self.var(x)
        return mean, log_var

class Decoder1(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, hidden_layers, y_dim, act_type, act_args):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        dims_arr = np.linspace(latent_dim + y_dim, hidden_dim, hidden_layers+1, dtype=int)
        
        for i, (in_size, out_size) in enumerate(zip(dims_arr[:-1], dims_arr[1:])):
            self.hidden.add_module(
                name='Linear_'+str(i),
                module=nn.Linear(in_size, out_size)
            )
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = self.hidden(x)
        generated_x = self.hidden_to_out(x)
        return generated_x

class CVAE1(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args={}):
        '''
        Args:
            input_dim: A integer indicating the size of input.
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder1(input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args)
        self.decoder = Decoder1(latent_dim, hidden_dim, input_dim, hidden_layers, y_dim, act_type, act_args)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var, z


def kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
   
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)

    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl

def calculate_loss(x, reconstructed_x, mu, log_var, weight, mc_kl_loss):
    # reconstruction loss
    rcl = F.mse_loss(reconstructed_x, x,  reduction='mean')
    # kl divergence loss

    if mc_kl_loss == True:
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        kld = kl_divergence(z, mu, std).sum()
    else:
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        #kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        #kld = -torch.sum(1+ log_var - mu.pow(2) - log_var.exp())
    
    rcl=rcl*10000
    kld=kld*10000
    #print(rcl, kld)
    
    return rcl, kld * weight
    #return rcl, kld


# ### CONFIG

# In[38]:

CONFIG = {}

CONFIG['gpu'] = True
CONFIG['params'] = {}
CONFIG['params']['seed'] = 42
CONFIG['params']['split_ratio'] = 0.2
CONFIG['params']['input_dim'] = 600+108 # input dimmension
CONFIG['params']['hidden_dim'] = 512
CONFIG['params']['latent_dim'] = 64
CONFIG['params']['hidden_layers'] = 3
CONFIG['params']['y_dim'] = 1
CONFIG['params']['batch_size'] = 512
CONFIG['params']['n_epochs'] = 4500
CONFIG['params']['lr'] = 2e-4
CONFIG['params']['final_decay'] = 0.2
CONFIG['params']['weight_decay'] = 0.001
CONFIG['params']['verbosity'] = 10
CONFIG['params']['kl_weight'] = 1e-6
CONFIG['params']['mc_kl_loss'] = False
CONFIG['params']['act_fn'] = "ELU"

CONFIG['data_x_path'] = 'MP_data_csv/perov_5_raw_train_dist_mat.pt'
CONFIG['data_ead_path'] = 'MP_data_csv/perov_5_raw_train_ead_mat.pt'
CONFIG['composition_path'] = 'MP_data_csv/perov_5_raw_train_composition_mat.pt'
CONFIG['cell_path'] = 'MP_data_csv/perov_5_raw_train_cell_mat.pt'
CONFIG['data_y_path'] = "MP_data_csv/perov_5/raw_train/targets.csv"

CONFIG['unprocessed_path'] = 'MP_data_csv/perov_5_raw_train_unprocessed.txt'

CONFIG['model_path'] = 'saved_models/cvae_saved.pt'
CONFIG['model_path2'] = 'saved_models/cvae_saved_dict.pt'
CONFIG['scaler_path'] = 'saved_models/scaler.gz'

CONFIG['train_model'] = True
CONFIG['generate_samps'] = 5

CONFIG = dotdict(CONFIG)
CONFIG['descriptor'] = ['ead', 'distance']

# EAD params
CONFIG['L'] = 1
CONFIG['eta'] = [1, 20, 90]
CONFIG['Rs'] = np.arange(0, 10, step=0.2)
CONFIG['derivative'] = False
CONFIG['stress'] = False

CONFIG['all_neighbors'] = True
CONFIG['perturb'] = False
CONFIG['load_pos'] = False
CONFIG['cutoff'] = 20.0
CONFIG['offset_count'] = 3

# ### Trainer

class Trainer1():

    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG

        # check GPU availability & set device
        self.device = torch_device_select(self.CONFIG.gpu)

        # initialize
        self.create_data()
        self.initialize()

    def create_data(self):

        p = self.CONFIG.params
        
        unprocessed = set()
        with open(self.CONFIG.unprocessed_path, 'r') as f:
            for l in f.readlines():
                unprocessed.add(int(l))

        # load pt files
        dist_mat = torch.load(self.CONFIG.data_x_path ,map_location=torch.device('cpu')).to("cpu")
        ead_mat = torch.load(self.CONFIG.data_ead_path,map_location=torch.device('cpu')).to("cpu")
        composition_mat = torch.load(self.CONFIG.composition_path,map_location=torch.device('cpu')).to("cpu")
        cell_mat = torch.load(self.CONFIG.cell_path,map_location=torch.device('cpu')).to("cpu")

        # build index
        _ind = [i for i in range(dist_mat.shape[0]) if i not in unprocessed]
        indices = torch.tensor(_ind, dtype=torch.long).to("cpu")

        # select rows torch.Size([27136, 1])
        dist_mat = dist_mat[indices] # the torch.load needs the index in tensor format to convert the loaded file in a tensor.
        ead_mat = ead_mat[indices]
        composition_mat = composition_mat[indices]
        cell_mat = cell_mat[indices]

        # normalize composition
        sums = torch.sum(composition_mat, axis=1).view(-1,1)
        composition_mat = composition_mat / sums
        composition_mat = torch.cat((composition_mat, sums), dim=1)

        y = []
        with open(self.CONFIG.data_y_path, 'r') as f:
            for i, d in enumerate(f.readlines()):
                if i not in unprocessed:
                    y.append(float(d.split(',')[1]))

        data_y = np.reshape(np.array(y), (-1,1)) 

        data_y = torch.from_numpy(data_y)
        data_y = data_y.to(torch.float32)

        print(composition_mat.max()) # tensor(20.)        
        print(composition_mat.shape, cell_mat.shape, dist_mat.shape, ead_mat.shape) # torch.Size([27136, 101]) torch.Size([27136, 6]) torch.Size([27136, 1]) torch.Size([27136, 600])
        
        data_x = torch.cat((ead_mat/1000000, composition_mat, cell_mat, dist_mat, data_y), dim=1)      

        print( (ead_mat/1000000).max(), composition_mat.max(), dist_mat.max(), cell_mat.max(), data_y.max())         
        
        #mask = data_x[:, 600] <= 10
        #data_x = data_x[mask]

        data_x, composition_mat, lat_mat, data_y = data_x[:, 0:600], data_x[:, 600:701], data_x[:,701:708], data_x[:,708]

        print(data_x.max(), composition_mat.max(), lat_mat.max(), data_y.max())

        print('lat_mat', [100])
        cell = lat_mat[100, 0:6]

        print(cell)
        #print(cell.shape) # torch.Size([6])
        print("cell1_m: ", cell)  # cell1:  tensor([3.8627, 4.6770, 6.7244, 1.5247, 1.7004, 1.5864])
        cell[3:6] = cell[3:6] * 180 / np.pi
        print("cell2_m: ", cell) 
        print('aye_yo')

        # scale
        scaler = MinMaxScaler()
        scaler.fit(data_x)
        data_x = scaler.transform(data_x)
        joblib.dump(scaler, self.CONFIG.scaler_path) # save the scaler to be used for later purpose on testing data.

        data_x = torch.from_numpy(data_x)
        data_x = data_x.to(torch.float32)

        data_x = torch.cat((10*data_x,composition_mat,lat_mat), dim=1) 

        data_y = data_y.view(-1,1)  

        print(data_x.shape)
        print(data_y.shape) 
           
        # train/test split and create torch dataloader
        xtrain, xtest, ytrain, ytest = train_test_split(
            data_x, data_y, test_size=self.CONFIG.split_ratio, random_state=p.seed
        )

        print(xtrain.shape)
        print(xtest.shape)
    

        if not isinstance(xtrain, torch.Tensor):
            self.x_train = torch.tensor(xtrain, dtype=torch.float)
        else:
            self.x_train = xtrain
            
        if not isinstance(ytrain, torch.Tensor):
            self.y_train = torch.tensor(ytrain, dtype=torch.float)
        else:
            self.y_train = ytrain
            
        if not isinstance(xtest, torch.Tensor):
            self.x_test = torch.tensor(xtest, dtype=torch.float)
        else:
            self.x_test = xtest
            
        if not isinstance(ytest, torch.Tensor):
            self.y_test = torch.tensor(ytest, dtype=torch.float)
        else:
            self.y_test = ytest
        
        indices = ~torch.any(self.x_train.isnan(),dim=1)

        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices] # y_train is the condition

        #indices = ~torch.any(self.x_train[:,:601] > 10 ,dim=1)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]     
        
        indices = ~torch.any(self.x_test.isnan(),dim=1)
        print(indices) # tensor([True, True, True,  ..., True, True, True])

        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]
        #indices = ~torch.any(self.x_test[:,:601] > 10 ,dim=1)
        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]        

        self.train_loader = DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=p.batch_size, shuffle=True, drop_last=False
        )

        self.test_loader = DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=p.batch_size, shuffle=False, drop_last=False
        )

    def initialize(self):
        p = self.CONFIG.params

        # create model
        self.model = CVAE1(p.input_dim, p.hidden_dim, p.latent_dim, p.hidden_layers, p.y_dim, p.act_fn)
        self.model.to(self.device)

        print(self.model)

        # set up optimizer
        # gamma = (p.final_decay)**(1./p.n_epochs)
        scheduler_args = {"mode":"min", "factor":0.8, "patience":20, "min_lr":1e-7, "threshold":0.0001}
        self.optimizer = optim.AdamW(self.model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_args
        )

    def train(self, kld_w):
        p = self.CONFIG.params
        self.model.train()

        # loss of the epoch 
        rcl_loss = 0.
        kld_loss = 0.

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # forward
            reconstructed_x, z_mu, z_var, z_latent = self.model(x, y) #generated_x, z_mu, z_var, z

            rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, kld_w, p.mc_kl_loss)

            # backward
            combined_loss = rcl + kld

            combined_loss.backward()
            rcl_loss += rcl.item()
            kld_loss += kld.item()

            # update the weights
            self.optimizer.step()
        
        return rcl_loss, kld_loss
    
    def test(self, kld_w):

        p = self.CONFIG.params
        self.model.eval()

        # loss of the evaluation
        rcl_loss = 0.
        kld_loss = 0.

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):

                # x is the input data 708 dimmensional
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass , return generated_x, z_mu, z_var
                reconstructed_x, z_mu, z_var, z_latent = self.model(x, y) # generated_x, z_mu, z_var, z

                # loss
                rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, kld_w, p.mc_kl_loss)
                rcl_loss += rcl.item()
                kld_loss += kld.item()
        
        return rcl_loss, kld_loss

    def run(self):
        p = self.CONFIG.params
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_epoch = 0

        kld_w = p.kl_weight
        rcl_prev = 0.0

        kld_l = []
        rcl_l = []
        kll_l = []


        for e in range(p.n_epochs):
            tic = time.time()

            rcl_train_loss, kld_train_loss = self.train(kld_w)
            rcl_test_loss, kld_test_loss = self.test(kld_w)

            rcl_train_loss /= len(self.x_train)
            kld_train_loss /= len(self.x_train)
            train_loss = rcl_train_loss + kld_train_loss
            rcl_test_loss /= len(self.x_test)
            kld_test_loss /= len(self.x_test)
            test_loss = rcl_test_loss + kld_test_loss

            self.scheduler.step(train_loss)
            lr = self.scheduler.optimizer.param_groups[0]["lr"]

            if best_test_loss > test_loss:
                best_epoch = e
                best_test_loss = test_loss
                best_train_loss = train_loss
                model_best = copy.deepcopy(self.model)

                # Save the model such that it can be loaded further at line 496.
                torch.save(model_best, self.CONFIG.model_path)
                torch.save(model_best.state_dict(), self.CONFIG.model_path2)                
            
            elapsed_time = time.time() - tic
            epoch_out = f'Epoch {e:04d},  kld: {kld_w:.5f},  Train RCL: {rcl_train_loss:.5f}, Train KLD: {kld_train_loss:.5f}, Train: {train_loss:.5f}, Test RLC: {rcl_test_loss:.5f}, Test KLD: {kld_test_loss:.5f}, Test: {test_loss:.5f}, LR: {lr:.5f}, Time/Epoch (s): {elapsed_time:.3f}'
            
            kld_l.append(kld_w)
            rcl_l.append(rcl_train_loss)
            kll_l.append(kld_train_loss)

            if (e%50 == 0): # change from 50
                kld_w = kld_w*1.15 # - (err_t/1e3)
            
            if(kld_w>0.01):
                kld_w = 0.01

            if e % p.verbosity == 0:
                print(epoch_out)

        return best_epoch, best_train_loss, best_test_loss, kld_l, rcl_l, kll_l

# Training 

trainer = Trainer1(CONFIG)
print(trainer.x_train.shape, trainer.x_test.shape) # torch.Size([18709, 708]) torch.Size([6230, 708])

if CONFIG.train_model == False:
    print('trainer running wild 1')
    sol = trainer.run() # the model runs and computes the reconstructed outputs, futher the optimiser computes the loss and adjust the weihgts through backprop.

model = trainer.model # check inisile method of trainer, it assigns cvae to model.
model.load_state_dict(torch.load(CONFIG.model_path2,map_location=torch.device('cpu'))) # method Inherited from nn.Module
model.eval() # method Inherited from nn.Module, model.eval() basically switches off the torch.grad the rest is the same, now next is out = model(testing set)

x1_x_test = model(trainer.x_test.to(trainer.device), trainer.y_test.to(trainer.device))
x1_x_train = model(trainer.x_train.to(trainer.device), trainer.y_train.to(trainer.device))

x_data_first_train = trainer.x_train
x_data_first_test = trainer.x_test

x1_test_R, x1_test_C, x1_test_L = x1_x_test[0][:, 0:600]/10, x1_x_test[0][:, 600:701] , x1_x_test[0][:, 701:708]
x1_train_R, x1_train_C, x1_train_L = x1_x_train[0][:, 0:600]/10, x1_x_train[0][:, 600:701] , x1_x_train[0][:, 701:708]

y1_test = trainer.y_test
y1_train = trainer.y_train

x1_train_R = x1_train_R.detach()
x1_train_L = x1_train_L.detach()
x1_train_C = x1_train_C.detach()

x1_test_R = x1_test_R.detach()
x1_test_L = x1_test_L.detach()
x1_test_C = x1_test_C.detach()

y1_test = y1_test.detach()
y1_train = y1_train.detach()

# ____________________________________________

class Encoder2(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        dims_arr = np.linspace(hidden_dim, latent_dim, hidden_layers+1, dtype=int)[0:-1]
        dims_arr = np.concatenate(([input_dim + y_dim], dims_arr))
        
        for i, (in_size, out_size) in enumerate(zip(dims_arr[:-1], dims_arr[1:])):
            self.hidden.add_module(
                name='Linear_'+str(i),
                module=nn.Linear(in_size, out_size)
            )
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.mu = nn.Linear(dims_arr[-1], latent_dim)
        self.var = nn.Linear(dims_arr[-1], latent_dim)

    def forward(self, x):
        x = self.hidden(x)
        mean = self.mu(x)
        log_var = self.var(x)
        return mean, log_var

class Decoder2(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, hidden_layers, y_dim, act_type, act_args):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        dims_arr = np.linspace(latent_dim + y_dim, hidden_dim, hidden_layers+1, dtype=int)
        
        for i, (in_size, out_size) in enumerate(zip(dims_arr[:-1], dims_arr[1:])):
            self.hidden.add_module(
                name='Linear_'+str(i),
                module=nn.Linear(in_size, out_size)
            )
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = self.hidden(x)
        generated_x = self.hidden_to_out(x)
        return generated_x

class CVAE2(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args={}):
        '''
        Args:
            input_dim: A integer indicating the size of input.
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder2(input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args)
        self.decoder = Decoder2(latent_dim, hidden_dim, input_dim, hidden_layers, y_dim, act_type, act_args)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var, z

# ### CONFIG

# In[38]:

CONFIG2 = {}

CONFIG2['gpu'] = True
CONFIG2['params'] = {}
CONFIG2['params']['seed'] = 42
CONFIG2['params']['split_ratio'] = 0.2
CONFIG2['params']['input_dim'] = 101+7 # input dimmension
CONFIG2['params']['hidden_dim'] = 512
CONFIG2['params']['latent_dim'] = 64
CONFIG2['params']['hidden_layers'] = 3
CONFIG2['params']['y_dim'] = 600+1 # 601
CONFIG2['params']['batch_size'] = 512
CONFIG2['params']['n_epochs'] = 4500

CONFIG2['params']['lr'] = 2e-4
CONFIG2['params']['final_decay'] = 0.2
CONFIG2['params']['weight_decay'] = 0.001
CONFIG2['params']['verbosity'] = 10
CONFIG2['params']['kl_weight'] = 1e-6
CONFIG2['params']['mc_kl_loss'] = False
CONFIG2['params']['act_fn'] = "ELU"

CONFIG2['data_x_path'] = 'MP_data_csv/perov_5_raw_train_dist_mat.pt'
CONFIG2['data_ead_path'] = 'MP_data_csv/perov_5_raw_train_ead_mat.pt'
CONFIG2['composition_path'] = 'MP_data_csv/perov_5_raw_train_composition_mat.pt'
CONFIG2['cell_path'] = 'MP_data_csv/perov_5_raw_train_cell_mat.pt'
CONFIG2['data_y_path'] = "MP_data_csv/perov_5/raw_train/targets.csv"

CONFIG2['unprocessed_path'] = 'MP_data_csv/perov_5_raw_train_unprocessed.txt'

CONFIG2['model_path'] = 'saved_models/cvae_saved2.pt'
CONFIG2['model_path2'] = 'saved_models/cvae_saved_dict2.pt'
CONFIG2['scaler_path2'] = 'saved_models/scaler2.gz'

CONFIG2['train_model'] = True
CONFIG2['generate_samps'] = 5

CONFIG2 = dotdict(CONFIG2)
CONFIG2['descriptor'] = ['ead', 'distance']

# EAD params
CONFIG2['L'] = 1
CONFIG2['eta'] = [1, 20, 90]
CONFIG2['Rs'] = np.arange(0, 10, step=0.2)
CONFIG2['derivative'] = False
CONFIG2['stress'] = False

CONFIG2['all_neighbors'] = True
CONFIG2['perturb'] = False
CONFIG2['load_pos'] = False
CONFIG2['cutoff'] = 20.0
CONFIG2['offset_count'] = 3

# ### Trainer

class Trainer2():

    def __init__(self, CONFIG2, x1_test_R, x1_train_R, y1_test, y1_train ) -> None:
        self.CONFIG = CONFIG2

        # check GPU availability & set device
        self.device = torch_device_select(self.CONFIG.gpu)

        # initialize
        self.create_data(x1_test_R, x1_train_R, y1_test, y1_train )
        self.initialize()

    def create_data(self, x1_test_R, x1_train_R, y1_test, y1_train):

        p = self.CONFIG.params
        print(self.CONFIG.unprocessed_path)
        
        unprocessed = set()
        with open(self.CONFIG.unprocessed_path, 'r') as f:
            for l in f.readlines():
                unprocessed.add(int(l))

        # load pt files
        dist_mat = torch.load(self.CONFIG.data_x_path ,map_location=torch.device('cpu')).to("cpu")
        ead_mat = torch.load(self.CONFIG.data_ead_path,map_location=torch.device('cpu')).to("cpu")
        composition_mat = torch.load(self.CONFIG.composition_path,map_location=torch.device('cpu')).to("cpu")
        cell_mat = torch.load(self.CONFIG.cell_path,map_location=torch.device('cpu')).to("cpu")

        # build index
        _ind = [i for i in range(dist_mat.shape[0]) if i not in unprocessed]
        indices = torch.tensor(_ind, dtype=torch.long).to("cpu")

        # select rows torch.Size([27136, 1])
        dist_mat = dist_mat[indices] # the torch.load needs the index in tensor format to convert the loaded file in a tensor.
        ead_mat = ead_mat[indices]
        composition_mat = composition_mat[indices]
        cell_mat = cell_mat[indices]

        # normalize composition
        sums = torch.sum(composition_mat, axis=1).view(-1,1)
        composition_mat = composition_mat / sums
        composition_mat = torch.cat((composition_mat, sums), dim=1)

        y = []
        with open(self.CONFIG.data_y_path, 'r') as f:
            for i, d in enumerate(f.readlines()):
                if i not in unprocessed:
                    y.append(float(d.split(',')[1]))

        data_y = np.reshape(np.array(y), (-1,1)) 

        data_y = torch.from_numpy(data_y)
        data_y = data_y.to(torch.float32)

        # ______________

        data_x = torch.cat((ead_mat/1000000, composition_mat, cell_mat, dist_mat, data_y), dim=1)             
        data_x, composition_mat, lat_mat, data_y = data_x[:, 0:600], data_x[:, 600:701], data_x[:,701:708], data_x[:,708]

        # ___________

        # scale

        #scaler2 = MinMaxScaler()
        #scaler2.fit(composition_mat)
        #composition_mat = scaler2.transform(composition_mat)
        #joblib.dump(scaler2, self.CONFIG.scaler_path2) # save the scaler to be used for later purpose on testing data.

        #composition_mat = torch.from_numpy(composition_mat)
        #composition_mat = composition_mat.to(torch.float32)

        comp1, comp2 = composition_mat[:,0:100], composition_mat[:,100]
        #print(comp1[0])
        #print(comp2[0])
        comp1 = 10*comp1
        composition_mat = torch.cat((comp1, comp2.view(-1,1)), dim=1)

        print(composition_mat[0])

        data_x = torch.cat((composition_mat, lat_mat), dim=1) 

        print('hello',data_x[0])

        data_y = data_y.view(-1,1)

        # train/test split and create torch dataloader
        xtrain, xtest, ytrain, ytest = train_test_split(
            data_x, data_y, test_size=self.CONFIG.split_ratio, random_state=p.seed
        )

        print('penum',data_x.shape)

        ytrain = torch.cat((x1_train_R, y1_train), dim=1)
        ytest = torch.cat((x1_test_R, y1_test), dim=1)   

        if not isinstance(xtrain, torch.Tensor):
            self.x_train = torch.tensor(xtrain, dtype=torch.float)
        else:
            self.x_train = xtrain
            
        if not isinstance(ytrain, torch.Tensor):
            self.y_train = torch.tensor(ytrain, dtype=torch.float)
        else:
            self.y_train = ytrain
            
        if not isinstance(xtest, torch.Tensor):
            self.x_test = torch.tensor(xtest, dtype=torch.float)
        else:
            self.x_test = xtest
            
        if not isinstance(ytest, torch.Tensor):
            self.y_test = torch.tensor(ytest, dtype=torch.float)
        else:
            self.y_test = ytest
        
        indices = ~torch.any(self.x_train.isnan(),dim=1)


        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices] # y_train is the condition

        #indices = ~torch.any(self.x_train[:,:601] > 10 ,dim=1)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]    
        
        indices = ~torch.any(self.x_test.isnan(),dim=1)
        print(indices) # tensor([True, True, True,  ..., True, True, True])

        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]
        #indices = ~torch.any(self.x_test[:,:601] > 10 ,dim=1)
        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]  


        self.train_loader = DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=p.batch_size, shuffle=True, drop_last=False
        )

        self.test_loader = DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=p.batch_size, shuffle=False, drop_last=False
        )

    def initialize(self):
        p = self.CONFIG.params

        # create model
        self.model = CVAE2(p.input_dim, p.hidden_dim, p.latent_dim, p.hidden_layers, p.y_dim, p.act_fn)
        self.model.to(self.device)

        print(self.model)

        # set up optimizer
        # gamma = (p.final_decay)**(1./p.n_epochs)
        scheduler_args = {"mode":"min", "factor":0.8, "patience":20, "min_lr":1e-7, "threshold":0.0001}
        self.optimizer = optim.AdamW(self.model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_args
        )

    def train(self, kld_w):
        p = self.CONFIG.params
        self.model.train()

        # loss of the epoch 
        rcl_loss = 0.
        kld_loss = 0.

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # forward
            reconstructed_x, z_mu, z_var, z_latent = self.model(x, y) #generated_x, z_mu, z_var, z

            rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, kld_w, p.mc_kl_loss)

            # backward
            combined_loss = rcl + kld

            combined_loss.backward()

            rcl_loss += rcl.item()
            kld_loss += kld.item()

            # update the weights
            self.optimizer.step()
        
        return rcl_loss, kld_loss
    
    def test(self, kld_w):

        p = self.CONFIG.params
        self.model.eval()

        # loss of the evaluation
        rcl_loss = 0.
        kld_loss = 0.

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):

                # x is the input data 708 dimmensional
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass , return generated_x, z_mu, z_var
                reconstructed_x, z_mu, z_var, z_latent = self.model(x, y) # generated_x, z_mu, z_var, z

                # loss
                rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, kld_w, p.mc_kl_loss)
                rcl_loss += rcl.item()
                kld_loss += kld.item()
        
        return rcl_loss, kld_loss

    def run(self):
        p = self.CONFIG.params
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_epoch = 0

        kld_w = p.kl_weight
        rcl_prev = 0.0

        kld_l = []
        rcl_l = []
        kll_l = []


        for e in range(p.n_epochs):
            tic = time.time()

            rcl_train_loss, kld_train_loss = self.train(kld_w)
            rcl_test_loss, kld_test_loss = self.test(kld_w)

            rcl_train_loss /= len(self.x_train)
            kld_train_loss /= len(self.x_train)
            train_loss = rcl_train_loss + kld_train_loss
            rcl_test_loss /= len(self.x_test)
            kld_test_loss /= len(self.x_test)
            test_loss = rcl_test_loss + kld_test_loss

            self.scheduler.step(train_loss)
            lr = self.scheduler.optimizer.param_groups[0]["lr"]

            if best_test_loss > test_loss:
                best_epoch = e
                best_test_loss = test_loss
                best_train_loss = train_loss
                model_best = copy.deepcopy(self.model)

                # Save the model such that it can be loaded further at line 496.
                torch.save(model_best, self.CONFIG.model_path)
                torch.save(model_best.state_dict(), self.CONFIG.model_path2)                
            
            elapsed_time = time.time() - tic
            epoch_out = f'Epoch {e:04d},  kld: {kld_w:.5f},  Train RCL: {rcl_train_loss:.5f}, Train KLD: {kld_train_loss:.5f}, Train: {train_loss:.5f}, Test RLC: {rcl_test_loss:.5f}, Test KLD: {kld_test_loss:.5f}, Test: {test_loss:.5f}, LR: {lr:.5f}, Time/Epoch (s): {elapsed_time:.3f}'
            
            kld_l.append(kld_w)
            rcl_l.append(rcl_train_loss)
            kll_l.append(kld_train_loss)

            if (e%50 == 0): # change from 50
                kld_w = kld_w*1.15 # - (err_t/1e3)
            
            if(kld_w>0.01):
                kld_w = 0.01

            if e % p.verbosity == 0:
                print(epoch_out)

        return best_epoch, best_train_loss, best_test_loss, kld_l, rcl_l, kll_l

# Training 

trainer = Trainer2(CONFIG2,x1_test_R, x1_train_R, y1_test, y1_train)
print(trainer.x_train.shape, trainer.x_test.shape) # torch.Size([8517, 101]) torch.Size([2839, 101])

if CONFIG2.train_model == False:
    print('trainer running wild 2')
    sol = trainer.run() # the model runs and computes the reconstructed outputs, futher the optimiser computes the loss and adjust the weihgts through backprop.

model = trainer.model # check inisile method of trainer, it assigns cvae to model.
model.load_state_dict(torch.load(CONFIG2.model_path2,map_location=torch.device('cpu'))) # method Inherited from nn.Module
model.eval() # method Inherited from nn.Module, model.eval() basically switches off the torch.grad the rest is the same, now next is out = model(testing set)

x2_x_test = model(trainer.x_test.to(trainer.device), trainer.y_test.to(trainer.device))
x2_x_train = model(trainer.x_train.to(trainer.device), trainer.y_train.to(trainer.device))

x2_test_C_1, x2_test_C_2, x2_test_L = x2_x_test[0][:, 0:100], x2_x_test[0][:,100], x2_x_test[0][:, 101:108]
x2_train_C_1, x2_train_C_2, x2_train_L = x2_x_train[0][:, 0:100],x2_x_train[0][:, 100], x2_x_train[0][:, 101:108]

x2_test_C = torch.cat((x2_test_C_1/10, x2_test_C_2.view(-1,1)), dim=1)
x2_train_C = torch.cat((x2_train_C_1/10, x2_train_C_2.view(-1,1)), dim=1)

y2_test = trainer.y_test
y2_train = trainer.y_train

print(y2_test.shape) # from previous model

x2_train_L = x2_train_L.detach()
x2_train_C = x2_train_C.detach()

x2_test_L = x2_test_L.detach()
x2_test_C = x2_test_C.detach()

y2_test = y2_test.detach()
y2_train = y2_train.detach()

# ____________________________________________

class Encoder3(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()
        dims_arr = np.linspace(hidden_dim, latent_dim, hidden_layers+1, dtype=int)[0:-1]
        dims_arr = np.concatenate(([input_dim + y_dim], dims_arr))
        
        for i, (in_size, out_size) in enumerate(zip(dims_arr[:-1], dims_arr[1:])):
            self.hidden.add_module(
                name='Linear_'+str(i),
                module=nn.Linear(in_size, out_size)
            )
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.mu = nn.Linear(dims_arr[-1], latent_dim)
        self.var = nn.Linear(dims_arr[-1], latent_dim)

    def forward(self, x):
        x = self.hidden(x)
        mean = self.mu(x)
        log_var = self.var(x)
        return mean, log_var

class Decoder3(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, hidden_layers, y_dim, act_type, act_args):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.hidden = nn.Sequential()


        dims_arr = np.linspace(latent_dim + y_dim, hidden_dim, hidden_layers+1, dtype=int)

        #print(dims_arr) # [734 660 586 512]
        
        for i, (in_size, out_size) in enumerate(zip(dims_arr[:-1], dims_arr[1:])):
            self.hidden.add_module(
                name='Linear_'+str(i),
                module=nn.Linear(in_size, out_size)
            )
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = self.hidden(x)
        generated_x = self.hidden_to_out(x)
        return generated_x

class CVAE3(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args={}):
        '''
        Args:
            input_dim: A integer indicating the size of input.
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder3(input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args)
        self.decoder = Decoder3(latent_dim, hidden_dim, input_dim, hidden_layers, y_dim, act_type, act_args)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var, z

# ### CONFIG

# In[38]:

CONFIG3 = {}

CONFIG3['gpu'] = True
CONFIG3['params'] = {}
CONFIG3['params']['seed'] = 42
CONFIG3['params']['split_ratio'] = 0.2
CONFIG3['params']['input_dim'] = 7 # input dimmension (L)
CONFIG3['params']['hidden_dim'] = 512
CONFIG3['params']['latent_dim'] = 32
CONFIG3['params']['hidden_layers'] = 3
CONFIG3['params']['y_dim'] = 600+101+1 # 608 R + C + Y
CONFIG3['params']['batch_size'] = 512
CONFIG3['params']['n_epochs'] = 4500
CONFIG3['params']['lr'] = 2e-4
CONFIG3['params']['final_decay'] = 0.2
CONFIG3['params']['weight_decay'] = 0.001
CONFIG3['params']['verbosity'] = 10
CONFIG3['params']['kl_weight'] = 1e-6
CONFIG3['params']['mc_kl_loss'] = False
CONFIG3['params']['act_fn'] = "ELU"

CONFIG3['data_x_path'] = 'MP_data_csv/perov_5_raw_train_dist_mat.pt'
CONFIG3['data_ead_path'] = 'MP_data_csv/perov_5_raw_train_ead_mat.pt'
CONFIG3['composition_path'] = 'MP_data_csv/perov_5_raw_train_composition_mat.pt'
CONFIG3['cell_path'] = 'MP_data_csv/perov_5_raw_train_cell_mat.pt'
CONFIG3['data_y_path'] = "MP_data_csv/perov_5/raw_train/targets.csv"

CONFIG3['unprocessed_path'] = 'MP_data_csv/perov_5_raw_train_unprocessed.txt'

CONFIG3['model_path'] = 'saved_models/cvae_saved3.pt'
CONFIG3['model_path3'] = 'saved_models/cvae_saved_dict3.pt'
CONFIG3['scaler_path3'] = 'saved_models/scaler3.gz'

CONFIG3['train_model'] = True
CONFIG3['generate_samps'] = 5

CONFIG3 = dotdict(CONFIG3)
CONFIG3['descriptor'] = ['ead', 'distance']

# EAD params
CONFIG3['L'] = 1
CONFIG3['eta'] = [1, 20, 90]
CONFIG3['Rs'] = np.arange(0, 10, step=0.2)
CONFIG3['derivative'] = False
CONFIG3['stress'] = False

CONFIG3['all_neighbors'] = True
CONFIG3['perturb'] = False
CONFIG3['load_pos'] = False
CONFIG3['cutoff'] = 20.0
CONFIG3['offset_count'] = 3

# ### Trainer

class Trainer3():

    def __init__(self, CONFIG3, x2_test_C, x2_train_C, y2_test, y2_train ) -> None:
        self.CONFIG = CONFIG3

        # check GPU availability & set device
        self.device = torch_device_select(self.CONFIG.gpu)

        # initialize
        self.create_data(x2_test_C, x2_train_C, y2_test, y2_train)
        self.initialize()

    def create_data(self, x2_test_C, x2_train_C, y2_test, y2_train ):

        p = self.CONFIG.params
        
        unprocessed = set()
        with open(self.CONFIG.unprocessed_path, 'r') as f:
            for l in f.readlines():
                unprocessed.add(int(l))

        # load pt files
        dist_mat = torch.load(self.CONFIG.data_x_path ,map_location=torch.device('cpu')).to("cpu")
        ead_mat = torch.load(self.CONFIG.data_ead_path,map_location=torch.device('cpu')).to("cpu")
        composition_mat = torch.load(self.CONFIG.composition_path,map_location=torch.device('cpu')).to("cpu")
        cell_mat = torch.load(self.CONFIG.cell_path,map_location=torch.device('cpu')).to("cpu")

        # build index
        _ind = [i for i in range(dist_mat.shape[0]) if i not in unprocessed]
        indices = torch.tensor(_ind, dtype=torch.long).to("cpu")

        # select rows torch.Size([27136, 1])
        dist_mat = dist_mat[indices] # the torch.load needs the index in tensor format to convert the loaded file in a tensor.
        ead_mat = ead_mat[indices]
        composition_mat = composition_mat[indices]
        cell_mat = cell_mat[indices]

        # normalize composition
        sums = torch.sum(composition_mat, axis=1).view(-1,1)
        composition_mat = composition_mat / sums
        composition_mat = torch.cat((composition_mat, sums), dim=1)

        y = []
        with open(self.CONFIG.data_y_path, 'r') as f:
            for i, d in enumerate(f.readlines()):
                if i not in unprocessed:
                    y.append(float(d.split(',')[1]))

        data_y = np.reshape(np.array(y), (-1,1)) 

        data_y = torch.from_numpy(data_y)
        data_y = data_y.to(torch.float32)

        #______________

        data_x = torch.cat((ead_mat/1000000, composition_mat, cell_mat, dist_mat, data_y), dim=1)             
        data_x, composition_mat, lat_mat, data_y = data_x[:, 0:600], data_x[:, 600:701], data_x[:,701:708], data_x[:,708]

        #_________________
        
        # scale
        scaler3 = MinMaxScaler()
        scaler3.fit(lat_mat)
        lat_mat = scaler3.transform(lat_mat)
        joblib.dump(scaler3, self.CONFIG.scaler_path3) # save the scaler to be used for later purpose on testing data.

        lat_mat = torch.from_numpy(lat_mat)
        lat_mat = lat_mat.to(torch.float32)

        data_x = 5*lat_mat #torch.cat((data_lattice, composition_mat), dim=1) 

        data_y = data_y.view(-1,1)

        # train/test split and create torch dataloader
        xtrain, xtest, ytrain, ytest = train_test_split(
            data_x, data_y, test_size=self.CONFIG.split_ratio, random_state=p.seed
        )   

        ytrain = torch.cat((x1_train_R, y1_train), dim=1)
        ytest = torch.cat((x1_test_R, y1_test), dim=1)  

        x_tr = xtrain
        x_ts = xtest

        y_tr = torch.cat((y2_train, x2_train_C), dim=1)
        y_ts = torch.cat((y2_test, x2_test_C), dim=1)   
        
        xtrain, xtest, ytrain, ytest = x_tr, x_ts, y_tr, y_ts 

        if not isinstance(xtrain, torch.Tensor):
            self.x_train = torch.tensor(xtrain, dtype=torch.float)
        else:
            self.x_train = xtrain
            
        if not isinstance(ytrain, torch.Tensor):
            self.y_train = torch.tensor(ytrain, dtype=torch.float)
        else:
            self.y_train = ytrain
            
        if not isinstance(xtest, torch.Tensor):
            self.x_test = torch.tensor(xtest, dtype=torch.float)
        else:
            self.x_test = xtest
            
        if not isinstance(ytest, torch.Tensor):
            self.y_test = torch.tensor(ytest, dtype=torch.float)
        else:
            self.y_test = ytest
        
        indices = ~torch.any(self.x_train.isnan(),dim=1)

        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices] # y_train is the condition

        #indices = ~torch.any(self.x_train[:,:601] > 10 ,dim=1)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]     
        
        indices = ~torch.any(self.x_test.isnan(),dim=1)
        print(indices) # tensor([True, True, True,  ..., True, True, True])

        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]
        #indices = ~torch.any(self.x_test[:,:601] > 10 ,dim=1)
        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]        

        self.train_loader = DataLoader(
            TensorDataset(self.x_train, self.y_train),
            batch_size=p.batch_size, shuffle=True, drop_last=False
        )

        self.test_loader = DataLoader(
            TensorDataset(self.x_test, self.y_test),
            batch_size=p.batch_size, shuffle=False, drop_last=False
        )

    def initialize(self):
        p = self.CONFIG.params

        # create model
        self.model = CVAE3(p.input_dim, p.hidden_dim, p.latent_dim, p.hidden_layers, p.y_dim, p.act_fn)
        self.model.to(self.device)

        print(self.model)

        # set up optimizer
        # gamma = (p.final_decay)**(1./p.n_epochs)
        scheduler_args = {"mode":"min", "factor":0.8, "patience":20, "min_lr":1e-7, "threshold":0.0001}
        self.optimizer = optim.AdamW(self.model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_args
        )

    def train(self, kld_w):
        p = self.CONFIG.params
        self.model.train()

        # loss of the epoch 
        rcl_loss = 0.
        kld_loss = 0.

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # forward
            reconstructed_x, z_mu, z_var, z_latent = self.model(x, y) #generated_x, z_mu, z_var, z

            rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, kld_w, p.mc_kl_loss)

            # backward
            combined_loss = rcl + kld

            combined_loss.backward()

            rcl_loss += rcl.item()
            kld_loss += kld.item()

            # update the weights
            self.optimizer.step()
        
        return rcl_loss, kld_loss
    
    def test(self, kld_w):

        p = self.CONFIG.params
        self.model.eval()

        # loss of the evaluation
        rcl_loss = 0.
        kld_loss = 0.

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):

                # x is the input data 708 dimmensional
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass , return generated_x, z_mu, z_var
                reconstructed_x, z_mu, z_var, z_latent = self.model(x, y) # generated_x, z_mu, z_var, z

                # loss
                rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var, kld_w, p.mc_kl_loss)
                rcl_loss += rcl.item()
                kld_loss += kld.item()
        
        return rcl_loss, kld_loss

    def run(self):
        p = self.CONFIG.params
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_epoch = 0

        kld_w = p.kl_weight
        rcl_prev = 0.0

        kld_l = []
        rcl_l = []
        kll_l = []


        for e in range(p.n_epochs):
            tic = time.time()

            rcl_train_loss, kld_train_loss = self.train(kld_w)
            rcl_test_loss, kld_test_loss = self.test(kld_w)

            rcl_train_loss /= len(self.x_train)
            kld_train_loss /= len(self.x_train)
            train_loss = rcl_train_loss + kld_train_loss
            rcl_test_loss /= len(self.x_test)
            kld_test_loss /= len(self.x_test)
            test_loss = rcl_test_loss + kld_test_loss

            self.scheduler.step(train_loss)
            lr = self.scheduler.optimizer.param_groups[0]["lr"]

            if best_test_loss > test_loss:
                best_epoch = e
                best_test_loss = test_loss
                best_train_loss = train_loss
                model_best = copy.deepcopy(self.model)

                # Save the model such that it can be loaded further at line 496.
                torch.save(model_best, self.CONFIG.model_path)
                torch.save(model_best.state_dict(), self.CONFIG.model_path3)                
            
            elapsed_time = time.time() - tic
            epoch_out = f'Epoch {e:04d},  kld: {kld_w:.5f},  Train RCL: {rcl_train_loss:.5f}, Train KLD: {kld_train_loss:.5f}, Train: {train_loss:.5f}, Test RLC: {rcl_test_loss:.5f}, Test KLD: {kld_test_loss:.5f}, Test: {test_loss:.5f}, LR: {lr:.5f}, Time/Epoch (s): {elapsed_time:.3f}'
            
            kld_l.append(kld_w)
            rcl_l.append(rcl_train_loss)
            kll_l.append(kld_train_loss)

            if (e%50 == 0): # change from 50
                kld_w = kld_w*1.15 # - (err_t/1e3)
            
            if(kld_w>0.01):
                kld_w = 0.01

            if e % p.verbosity == 0:
                print(epoch_out)

        return best_epoch, best_train_loss, best_test_loss, kld_l, rcl_l, kll_l

# Training 

trainer = Trainer3(CONFIG3, x2_test_C, x2_train_C, y2_test, y2_train)
print(trainer.x_train.shape, trainer.x_test.shape) # torch.Size([18709, 708]) torch.Size([6230, 708])

if CONFIG.train_model == False:
    print('trainer running wild 3')
    sol = trainer.run() # the model runs and computes the reconstructed outputs, futher the optimiser computes the loss and adjust the weihgts through backprop.

model = trainer.model # check inisile method of trainer, it assigns cvae to model.
model.load_state_dict(torch.load(CONFIG3.model_path3,map_location=torch.device('cpu'))) # method Inherited from nn.Module
model.eval() # method Inherited from nn.Module, model.eval() basically switches off the torch.grad the rest is the same, now next is out = model(testing set)

x_test3 = model(trainer.x_test.to(trainer.device), trainer.y_test.to(trainer.device))
x_train3 = model(trainer.x_train.to(trainer.device), trainer.y_train.to(trainer.device)) 

print(x_test3[0].shape) # torch.Size([2839, 7])
print(x_train3[0].shape) # torch.Size([8517, 7]) 

x_ts = torch.cat((x1_x_test[0][:,0:600], x2_x_test[0][:,0:101], x_test3[0]), dim = 1)
x_tr = torch.cat((x1_x_train[0][:,0:600], x2_x_train[0][:,0:101], x_train3[0]), dim = 1)

R_n_s, C_n_s, L_n_s = x1_x_test[0][:,0:600], x2_x_test[0][:,0:101], x_test3[0]/5

print(x_ts.shape) # torch.Size([2839, 708])
print(x_tr.shape) # torch.Size([8517, 708])

scaler1 = joblib.load(CONFIG.scaler_path)

scaler2 = joblib.load(CONFIG2.scaler_path2)

scaler3 = joblib.load(CONFIG3.scaler_path3)

R_s = scaler1.inverse_transform(R_n_s.cpu().data.numpy())
R_s = torch.tensor(R_s, dtype=torch.float)

C_s = scaler2.inverse_transform(C_n_s.cpu().data.numpy())
C_s = torch.tensor(C_s, dtype=torch.float)

L_s = scaler3.inverse_transform(L_n_s.cpu().data.numpy())
L_s = torch.tensor(L_s, dtype=torch.float)


cell = L_s[100, 0:6]

print(cell)
print(C_s[100])

#print(cell.shape) # torch.Size([6])
print("cell1_m: ", cell)  # cell1:  tensor([3.8627, 4.6770, 6.7244, 1.5247, 1.7004, 1.5864])
cell[3:6] = cell[3:6] * 180 / np.pi
print("cell2_m: ", cell) 

x1_R, x1_C, x1_L = x_data_first_test[:, 0:600], x_data_first_train[:, 600:701] , x_data_first_train[:, 701:708]

cell = x1_L[100, 0:6]

print('C_train', 100*x1_C[100])

#print(cell.shape) # torch.Size([6])
print("cell1_t: ", cell)  # cell1:  tensor([3.8627, 4.6770, 6.7244, 1.5247, 1.7004, 1.5864])
cell[3:6] = cell[3:6] * 180 / np.pi
print("cell2_t: ", cell) 

#print(x_ts[0])





