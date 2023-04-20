#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#CUDA_VISIBLE_DEVICES is used to specify which GPUs should be visible to a CUDA application via index of GPU.


# In[ ]:


#Downloading data if it does not previously exsist.

import git #The git library is used to make API calls and interact with git repositories.
from os import path #Path is used to facilitate checking if a file/directory exsists.

if not path.exists('go-dataset-master'): # Checks if data does not previously exsist.
    
    print('Cloning')
    
    git.Repo.clone_from('https://github.com/featurecat/go-dataset.git', '', branch='master') 
    
    #API call which clones repository. 


# In[ ]:


from utils import get_path_from_file

import glob #Module used to find all pathnames that match specified pattern.
from pyunpack import Archive #Library used to extract 7z files.


if not path.exists('go-dataset-master/10k/10k.7z_extracted'): #Checks if data has previously been extracted.
    
    paths_to_7z = glob.glob('go-dataset-master/*/*', recursive=True) #Finds all 7z files.
    
    for path_7z in paths_to_7z: 
        
        path_from = get_path_from_file(path_7z)
        Archive(path_7z).extractall(path_from) 
        
        #Extracts all 7z files in their corresponding parent directory.


# In[ ]:


import torch
import numpy as np #NumPy is a library that supports multidimensional arrays and is commonly used alongside PyTorch.


# In[ ]:


paths_10k = glob.glob('go-dataset-master/10k/*/*/*') #Creates a list with paths to all SGF files in directory.
paths_5k = glob.glob('go-dataset-master/5k/*/*/*')
paths_1d = glob.glob('go-dataset-master/1d/*/*/*')


# In[ ]:


samples = np.random.normal(30, 170, 1000000) #Normal distribution used to generate game lengths.

truncated_samples = []
probability_dict ={k:0 for k in range(0,330)} 

for sample in samples:
    
    if sample > 0 and sample < 330: #Will prevent showing the network excessively long (and uncommon) games states.
        
        sample = np.floor(sample) 
        truncated_samples.append(sample)
        probability_dict[sample]+=1 
        
        
samples_len = len(truncated_samples)
probability_dict = {k: v / samples_len for k, v in probability_dict.items()} # Converting to probabilities.


# In[ ]:


n_gpus = torch.cuda.device_count() 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Checks if GPU is available. 

if n_gpus==1: 
    device = torch.device(0) #If one GPU is available sets the variable decive to this so that PyTorch tensors and 
                             #models can be easily transferred to GPU memory.

print(device) #Displays the device being used.


# In[ ]:


from model import GoModel
import torch.optim as optim
from train import train

model = GoModel() #Instantiating object 

epochs = 15
paths = paths_1d 

#Stochastic optimization method similar to gradient descent but computes individual adaptive learning for weights.
optimizer = optim.AdamW(model.parameters(), lr=0.003)

#Increases the learning rate for 30% of training and decreases it for the rest, has been shown to improve training.
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 0.003, epochs= epochs, steps_per_epoch = len(paths)//400)


# In[ ]:


train('1d_v1', model, epochs, optimizer, scheduler, paths, device,  None)


# In[ ]:




