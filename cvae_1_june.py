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



class Encoder(nn.Module):
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
                module= nn.Linear(in_size, out_size)
            )
            print('in_size = ', in_size)
            print('out_size = ', out_size)
            self.hidden.add_module(
                name='Act_'+str(i), 
                module=getattr(nn, act_type)(**act_args)
            )

        self.mu = nn.Linear(dims_arr[-1], latent_dim)
        self.var = nn.Linear(dims_arr[-1], latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]
        #print('x', x.dtype)
        x = self.hidden(x)
        #print('x', x.dtype)
        # hidden is of shape [batch_size, hidden_dim]
        # latent parameters
        mean = self.mu(x)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(x)
        # log_var is of shape [batch_size, latent_dim]
        return mean, log_var

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''

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
            print('in_size2 = ', in_size)
            print('out_size2 = ', out_size)
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


class CVAE(nn.Module):
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

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, hidden_layers, y_dim, act_type, act_args)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, hidden_layers, y_dim, act_type, act_args)

    def forward(self, x, y):

        print(x.shape)
        print(y.shape)

        x = torch.cat((x, y), dim=1)

        print('hello')

        print(x.shape)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
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
CONFIG['params']['input_dim'] = 708 # input dimmension
CONFIG['params']['hidden_dim'] = 512
CONFIG['params']['latent_dim'] = 64
CONFIG['params']['hidden_layers'] = 3
CONFIG['params']['y_dim'] = 1
CONFIG['params']['batch_size'] = 512
CONFIG['params']['n_epochs'] = 5000
CONFIG['params']['lr'] = 2e-4
CONFIG['params']['final_decay'] = 0.2
CONFIG['params']['weight_decay'] = 0.001
CONFIG['params']['verbosity'] = 10
CONFIG['params']['kl_weight'] = 1e-6
CONFIG['params']['mc_kl_loss'] = False
CONFIG['params']['act_fn'] = "ELU"

#CONFIG['data_x_path'] = 'MP_data_csv/mp_20_raw_train_dist_mat.pt'
#CONFIG['data_ead_path'] = 'MP_data_csv/mp_20_raw_train_ead_mat.pt'
#CONFIG['composition_path'] = 'MP_data_csv/mp_20_raw_train_composition_mat.pt'
#CONFIG['cell_path'] = 'MP_data_csv/mp_20_raw_train_cell_mat.pt'
#CONFIG['data_y_path'] = "MP_data_csv/mp_20/raw_train/targets.csv"

#CONFIG['unprocessed_path'] = 'MP_data_csv/mp_20_raw_train_unprocessed.txt'

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


class Trainer():

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
        data_x = torch.cat((ead_mat/1000000, dist_mat, cell_mat, composition_mat, data_y), dim=1)        
        #print(data_x.shape) # torch.Size([27136, 607]) i.e coloumn [0:606]
        #print((data_x.max())) #tensor(999999986991104.) ; tensor(999999986991104.) 9999999827968

        #print(data_x.shape) # torch.Size([27136, 708])
        #print(data_x[0][600])

        # Create a boolean mask to identify the rows where the value in the specified column is equal to the drop value
        mask = data_x[:, 600] <= 10
        # Use the mask to index the tensor and select only the rows that do not have the drop value in the specified column
        data_x = data_x[mask]

        print('hello')
        print(data_x.shape)
        data_x, composition_mat, data_y = data_x[:, 0:607], data_x[:,607:708] , data_x[:,708] 
        print(data_x.shape)
        print(composition_mat.shape)

        exit()
        print(data_x[10][601:607]) #0,1,2...599,600,601

        # scale
        scaler = MinMaxScaler()
        scaler.fit(data_x)
        data_x = scaler.transform(data_x)
        joblib.dump(scaler, self.CONFIG.scaler_path) # save the scaler to be used for later purpose on testing data.

        
        comp1, comp2 = composition_mat[:, 0:100], composition_mat[:,100]
        comp1 = 5*(comp1.to(torch.float32))
        comp2 = 5*comp2.to(torch.float32).view(-1,1)


        composition_mat_add = torch.cat((comp1,comp2), dim=1) 

        data_x = torch.from_numpy(data_x)
        data_x = data_x.to(torch.float32)
        data_x = torch.cat((data_x,composition_mat_add), dim=1)  

        print(data_x.shape)
        print(data_x[10,601:607])
        print(data_x.max())  #tensor(20.)

        data_y = data_y.view(-1,1)
        print(data_y)

        print(data_y.max())  #tensor(1.0000002)

        # train/test split and create torch dataloader
        xtrain, xtest, ytrain, ytest = train_test_split(
            data_x, data_y, test_size=self.CONFIG.split_ratio, random_state=p.seed
        )

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

        indices = ~torch.any(self.x_train[:,:601] > 10 ,dim=1)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]     
        
        indices = ~torch.any(self.x_test.isnan(),dim=1)
        print(indices) # tensor([True, True, True,  ..., True, True, True])

        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]
        indices = ~torch.any(self.x_test[:,:601] > 10 ,dim=1)
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
        self.model = CVAE(p.input_dim, p.hidden_dim, p.latent_dim, p.hidden_layers, p.y_dim, p.act_fn)
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

            #err_t = (rcl_train_loss - rcl_prev)
            if (e%50 == 0): # change from 50
                kld_w = kld_w*1.15 # - (err_t/1e3)

            #if (e%200 == 0):
            #    kld_w = kld_w*10 # - (err_t/1e3)

            #if(kld_w<1e-12):
            #    kld_w = 1e-12

            #if(kld_w>0.00013):
            #    kld_w = 0.00013
            
            if(kld_w>0.01):
                kld_w = 0.01

            #rcl_prev = rcl_train_loss
            
            if e % p.verbosity == 0:
                print(epoch_out)

        return best_epoch, best_train_loss, best_test_loss, kld_l, rcl_l, kll_l

# Training 

trainer = Trainer(CONFIG)
print(trainer.x_train.shape, trainer.x_test.shape) # torch.Size([18709, 708]) torch.Size([6230, 708])

# print(trainer.x_train[0], trainer.x_train.isnan(), trainer.x_train.max())
# print('latent_dim', latent_dim)

if CONFIG.train_model == True:
    print('trainer running wild')
    sol = trainer.run() # the model runs and computes the reconstructed outputs, futher the optimiser computes the loss and adjust the weihgts through backprop.
    
# after trainer has finished running trainer.run ; the self.model is updated. 
# hence trainer.model has now the updated values if trainer.run is executed.

'''
x_t_1_l = len(sol[3])
x_t_1 = np.linspace(0, x_t_1_l, x_t_1_l) # 0 --- 101
y_t_1 = np.array(sol[3])
y_t_2 = np.array(sol[4])
y_t_3 = np.array(sol[5])
y_t_1 = y_t_1.tolist()
y_t_2 = y_t_2.tolist()
y_t_3 = y_t_3.tolist()
plt.semilogy(x_t_1,y_t_1) # A bar chart
plt.xlabel('epoch')
plt.ylabel('kld_w')
plt.show()
plt.semilogy(x_t_1,y_t_2) # A bar chart
plt.semilogy(x_t_1,y_t_3) # A bar chart
plt.xlabel('epoch')
plt.ylabel('Errors in rcl and kld')
plt.show()


model = trainer.model # check inisile method of trainer, it assigns cvae to model.
# Next we load the trained model.
model.load_state_dict(torch.load(CONFIG.model_path2,map_location=torch.device('cpu'))) # method Inherited from nn.Module
model.eval() # method Inherited from nn.Module, model.eval() basically switches off the torch.grad the rest is the same, now next is out = model(testing set)

# extract 1000 data points for pca analysis before re-scalling
x_pca = trainer.x_train[:500]
y_pca_t = trainer.y_train[:500]
x_pca2 = trainer.x_test[:500]
y_pca2_t = trainer.y_test[:500]

ranges = [(0, 0.5), (0.6, 1.0), (1.1, 1.5)]

y_pca = torch.zeros_like(y_pca_t)

for i, r in enumerate(ranges):
    indices = torch.where((y_pca_t >= r[0]) & (y_pca_t <= r[1]))
    y_pca[indices] = (r[0]+r[1])/2
    indices = torch.where((y_pca_t >= 1.5))
    y_pca[indices] = 1.5

y_pca2 = torch.zeros_like(y_pca2_t)

for i, r in enumerate(ranges):
    indices = torch.where((y_pca2_t >= r[0]) & (y_pca2_t <= r[1]))
    y_pca2[indices] = (r[0]+r[1])/2
    indices = torch.where((y_pca2_t >= 1.5))
    y_pca[indices] = 1.5


print('xx')
print(trainer.y_train.shape) # 8517
print(trainer.y_test.shape) # 2839
print(y_pca2.shape)
print('xx')

y_t_1 = y_pca2.cpu().detach().numpy()
x_t_1 = np.linspace(1, 500, 500) # 0 --- 101

x_t_1= x_t_1.tolist()
y_t_1= y_t_1.flatten().tolist()

plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('y_values_test')
plt.show()

y_t_1 = y_pca.cpu().detach().numpy()
x_t_1 = np.linspace(1, 500, 500) # 0 --- 101

x_t_1= x_t_1.tolist()
y_t_1= y_t_1.flatten().tolist()

plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('y_values_train')
plt.show()



# load the scaler
#scaler = load(open(CONFIG.scaler_path)) this won't work
scaler = joblib.load(CONFIG.scaler_path)
test_inp = np.array([[208.439621 , 231.262134 , 173.1262134, 169.1],[408.439621 , 431.262134 , 373.1262134, 369.1]])
test_inp = scaler.fit_transform(test_inp)

#y_pca = scaler.inverse_transform(y_pca.cpu().data.numpy())
#y_pca = torch.tensor(y_pca, dtype=torch.float, device=trainer.device)

#y_pca2 = scaler.inverse_transform(y_pca2.cpu().data.numpy())
#y_pca2 = torch.tensor(y_pca2, dtype=torch.float, device=trainer.device)

out = model(trainer.x_test[:100].to(trainer.device), trainer.y_test[:100].to(trainer.device))
out2 = model(trainer.x_train[:100].to(trainer.device), trainer.y_train[:100].to(trainer.device)) 

print(trainer.x_train[:100][0])
#print(out2[0][0])

print('flag1')
print('helli')

# inverse scalling of the output data.
scale = preprocessing.MinMaxScaler()
scale.min_,scale.scale_ = scaler.min_[0], scaler.scale_[0] #scale.min_,scale.scale_=scaler.min_[0],scaler.scale_[0]

# out1 starts ; i.e. on test data

out_l = out[0][:,0:607]
out_r = out[0][:,607:707]

print('xxx_tentacion_test')
print(trainer.x_test[0,607:707])
print(out_r[0])

out_r = out_r.cpu().detach().numpy()
trainer.x_test = trainer.x_test.cpu().detach().numpy()

x_t_1 = np.linspace(0, 99, 100) # 0 --- 101

y_f = np.mean(out_r, axis=0)
y_i = np.mean(trainer.x_test[0,607:707], axis=0)

y_t_1 = ((y_f-y_i)/y_i)*100

x_t_1= x_t_1.tolist()
y_t_1= y_t_1.tolist()

plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('percentage_error_test')
plt.show()

y_t_1 = ((y_f-y_i))
y_t_1= y_t_1.tolist()

plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('error')
plt.show()


# out2 starts ; i.e. on train data

#print(out_l.shape) #torch.Size([100, 607])
#print(out_r.shape) #torch.Size([100, 101])

out_l = out2[0][:,0:607]
out_r = out2[0][:,607:707]
#exit()

#err= (trainer.x_test[0,606:707] - out_r)
#err_p = 100*(err/(trainer.x_test[0,606:707]))
#print(err)
#print(err_p)

print('xxx_tentacion_train')
print(trainer.x_train[0,607:707])
print(out_r[0])

#err= (trainer.x_train[0,606:707] - out_r)
#err_p = 100*(err/(trainer.x_train[0,606:707]))
#print(err)
#print(err_p)

print('xxx_tentacion')

out_r = out_r.cpu().detach().numpy()
trainer.x_train = trainer.x_train.cpu().detach().numpy()

x_t_1 = np.linspace(0, 99, 100) # 0 --- 101

y_f = np.mean(out_r, axis=0)
y_i = np.mean(trainer.x_train[0,606:707], axis=0)
y_t_1 = ((y_f-y_i)/y_i)*100

x_t_1= x_t_1.tolist()
y_t_1= y_t_1.tolist()

plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('percentage_error_test')
plt.show()

y_t_1 = ((y_f-y_i))
y_t_1 = y_t_1.tolist()

print('game')
print(y_t_1)

plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('error')
plt.show()

out_r = out2[0]#[:,:]
out_r = out_r.cpu().detach().numpy()

a11 = trainer.x_train[10,:]
y_1 = trainer.y_train[10,:]
a12 = out_r[10]

a22 = trainer.x_train[50,:]
y_2 = trainer.y_train[50,:]
a21 = out_r[50]

out_r = out[0]#[:,:]

a33 = trainer.x_test[10,:]
y_3 = trainer.y_test[10,:]
a31 = out_r[50]

a44 = trainer.x_test[50,:]
y_4 = trainer.y_test[50,:]
a41 = out_r[50]

print(a11.shape)

#rev_x_need_scaling, composition_vec = a11[:CONFIG.params.input_dim-101], a11[:-102] 
#cell = rev_x_need_scaling[601:607]
#print(cell)

#print(y_1)
#print(a11)
#print(torch.tensor(a12))

#print(y_2)
#print(torch.tensor(a22))
#print(torch.tensor(a21))

print('____________________________________________')

print(y_3)
print(torch.tensor(a33))
print(torch.tensor(a31))

print(y_4)
print(torch.tensor(a44))
print(torch.tensor(a41))


out_r = out_r.cpu().detach().numpy()

x_t_1 = np.linspace(0, 708, 708) # 0 --- 101

y_t_1 = ((trainer.x_train[30,:]- out_r[30]))
y_t_1 = y_t_1.tolist()
plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('error_train_30')
plt.show()

out_r = out2[0]#[:,:]
out_r = out_r.cpu().detach().numpy()

x_t_1 = np.linspace(0, 708, 708) # 0 --- 101

y_t_1 = ((trainer.x_train[20,:]- out_r[20]))
y_t_1 = y_t_1.tolist()
plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('error_train_20')
plt.show()

out_r = out[0]#[:,:]
lat_z = out[3]
out_r = out_r.cpu().detach().numpy()

x_t_1 = np.linspace(0, 708, 708) # 0 --- 101

y_t_1 = ((trainer.x_test[40,:]- out_r[40]))
y_t_1 = y_t_1.tolist()

print('helli')


print('latent',lat_z[40].tolist())
print(lat_z[40].shape)
print(out_r[40].tolist())
print(out_r[40].shape)

print('x_test')

print(trainer.x_test[40,:].tolist())
print(trainer.x_test[40,:].shape)

print('helli')

#exit()

plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('error_test40')
plt.show()


out_r = out[0]#[:,:]
lat_z = out[3]
out_r = out_r.cpu().detach().numpy()

x_t_1 = np.linspace(0, 708, 708) # 0 --- 101

y_t_1 = ((trainer.x_test[10,:]- out_r[10]))
y_t_1 = y_t_1.tolist()


plt.bar(x_t_1,y_t_1,align='center') # A bar chart
plt.xlabel('Bins')
plt.ylabel('error_test10')
plt.show()


##### PCA #####

## Histogram end

###Add for loop to generate n samples
#print(CONFIG.generate_samps) # 5
print('flag2')


# Train

pca = PCA(n_components=2)
#x_reshaped = x_test[0:500].reshape(-1, 784)              # new shape is (500, 28*28) = (500, 784)
out2 = model(x_pca, y_pca)
# 10 out of 6230 output enteries all in out
#print(out2[0].shape) #torch.Size([10, 708])

y_color_map = y_pca#.to(trainer.device)
print('color',y_color_map.shape)
#print(y_color_map)

out2 = out2[3].detach().numpy()
#out2 = scaler.inverse_transform(out2[0].cpu().data.numpy())


#x_scaled = StandardScaler().fit_transform(out2)    # center and scale data (mean=0, std=1)
#x_transformed = pca.fit(x_scaled).transform(x_scaled)

plt.figure()

cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(out2[:, 0], out2[:, 1],
            s=20, alpha=.8 , cmap=cm, c= y_color_map) # cmap='Set1', c=y_test[0:500]

plt.title("PCA train data 1000 samples")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(sc)
plt.show()

# Testing on randomly generated samples.

pca = PCA(n_components=2)
#x_reshaped = x_test[0:500].reshape(-1, 784)              # new shape is (500, 28*28) = (500, 784)
out2 = model(x_pca2, y_pca2)
# 10 out of 6230 output enteries all in out
#print(out2[0].shape) #torch.Size([10, 708])

y_color_map = y_pca2#.to(trainer.device)
print('color',y_color_map.shape)
#print(y_color_map)

out2_lat = out2[3]
out2_gen = out2[0]
out2 = out2_lat.detach().numpy()
out2_gen = out2_gen.detach().numpy()

print('z_latent')
print(out2)
print(out2[10])
print(out2_gen[10])

#out2 = scaler.inverse_transform(out2[0].cpu().data.numpy())

#x_scaled = StandardScaler().fit_transform(out2)    # center and scale data (mean=0, std=1)
#x_transformed = pca.fit(x_scaled).transform(x_scaled)

plt.figure()
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(out2[:, 0], out2[:, 1],
            s=20, alpha=.8 , cmap=cm, c= y_color_map) # cmap='Set1', c=y_test[0:500]

plt.title("PCA testing data 1000 samples")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(sc)
plt.show()

pca = PCA(n_components=3)
#x_reshaped = x_test[0:500].reshape(-1, 784)              # new shape is (500, 28*28) = (500, 784)
out2 = model(x_pca, y_pca)
# 10 out of 6230 output enteries all in out
#print(out2[0].shape) #torch.Size([10, 708])

y_color_map = y_pca#.to(trainer.device)
print('color',y_color_map.shape)
#print(y_color_map)

out2 = out2[3].detach().numpy()
#out2 = scaler.inverse_transform(out2[0].cpu().data.numpy())


#x_scaled = StandardScaler().fit_transform(out2)    # center and scale data (mean=0, std=1)
#x_transformed = pca.fit(x_scaled).transform(x_scaled)

plt.figure()

cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(out2[:, 0], out2[:, 1],
            s=20, alpha=.8 , cmap=cm, c= y_color_map) # cmap='Set1', c=y_test[0:500]

plt.title("PCA train data 1000 samples")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(sc)
plt.show()

pca = PCA(n_components=3)
#x_reshaped = x_test[0:500].reshape(-1, 784)              # new shape is (500, 28*28) = (500, 784)
#out2 = model(trainer.x_train[:1000].to(trainer.device), trainer.y_train[:1000].to(trainer.device))
out2 = model(x_pca, y_pca)
# 10 out of 6230 output enteries all in out
#print(out2[0].shape) #torch.Size([10, 708])

#y_color_map = trainer.y_train[:1000].to(trainer.device)
y_color_map = y_pca#.to(trainer.device)
print('color',y_color_map.shape)
#print(y_color_map)

#out2 = scaler.inverse_transform(out2[0].cpu().data.numpy())
out2 = out2[3].detach().numpy()

#x_scaled = StandardScaler().fit_transform(out2)    # center and scale data (mean=0, std=1)
#x_transformed = pca.fit(x_scaled).transform(x_scaled)
ax = plt.axes(projection ="3d")

plt.figure()
cm = plt.cm.get_cmap('RdYlBu')
sc = ax.scatter3D(out2[:, 0], out2[:, 1], out2[:, 2],
            s=20, alpha=.8 , cmap=cm, c= y_color_map) # cmap='Set1', c=y_test[0:500]

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_ylabel('Principal Component 3')
plt.colorbar(sc)
plt.show()

'''