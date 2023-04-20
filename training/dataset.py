from torch.utils.data import Dataset 
import numpy as np
from preprocess import open_games, process_data
import torch

boardsize=19

class GoDataset(Dataset): #Dataset class with inheritance from PyTorch's Dataset class.
    
    def __init__(self, paths_list, test = True):
        self.test = test 
        opened_games, self.winners = open_games(paths_list) 
        self.tensor_input, self.ground_truth, self.colours  = process_data(opened_games)
        
        #print(f'{len(paths_list)-len(self.ground_truth)} file(s) in incorrect SGF format found and not added to dataset')
        
        
    def __len__(self): #Overrides length method (init and getitem are also overriden) and returns the size of the
                       #dataset.
            
        return len(self.ground_truth)
            
    def __getitem__(self, idx):
        
        colour = self.colours[idx] 
        colour_map = torch.full((boardsize, boardsize), colour) #The AI's turn is turned into a tensor full of either
                                                                #one's (black) or minus one's (white) to indicate how 
                                                                #it should play.
        #Corresponding input and output pairs.
        net_input = self.tensor_input[idx] 
        ground_truth = self.ground_truth[idx]
        
        winner = self.winners[idx] #The true outcome of the game is also used as a target for the value part of the 
                                   #neural network.
            
        if not self.test: #If the dataset is going to be used for training instead of validation, random rotations are  
                          #added to augment the training data.
                
            k = np.random.randint(4)
            net_input = torch.rot90(net_input, k)
            ground_truth = torch.rot90(ground_truth, k)
            
        net_input_black = net_input==1 #Input to the neural network is split into two channels, both full of ones and 
        net_input_white = net_input==-1 #zeros, which indicate black and white's pieces in diffrent channels.
        
        return torch.stack((net_input_black.int(), net_input_white.int(), colour_map)), ground_truth, torch.tensor(winner)
    