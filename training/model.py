import torch 
#PyTorch is the machine learning framework that will handle the neural network implementation and optimization.

import torch.nn as nn #The importing several frequently used PyTorch modules with a shortened name 
                      #so that the code is more readable.
import torch.nn.functional as F

boardsize = 19

class ResidualBlock(nn.Module): #The architecture of the neural network consists primarily of redisual blocks.
    
    def __init__(self, planes=256, kernel_size =3 , stride=1, bias = True): #Planes and kernel size same as AlphaGo.
        
        super().__init__() #Used to give access to methods and properties of a parent class.
        
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=kernel_size, #Convolutional layer with padding set to 
                        stride=stride, padding='same', bias=bias)       #'same' so output shape is the same.
        
        self.bn1 = nn.BatchNorm2d(planes) #Batch normalization.
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                        stride=stride, padding='same', bias=bias)
        
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x): #Convolutional then batch normalization and then rectified linear unit (twice).
        
        residual = x 
        
        out = self.conv1(x) 
        
        out = self.bn1(out)
        
        out = F.relu(out) 
        
        out = self.conv2(out)
        
        out = self.bn2(out)
        
        out += residual #Skip connection to avoid the vanishing gradient problem.
        
        out = F.relu(out)
        
        return out
    
class ValueHead(nn.Module): #Value head used to predict who the winner of the game is going to be.
    def __init__(self, planes_in = 256 ,planes=1, kernel_size =1 , stride=1, bias = True): #Kernel set to one as
                                                                                           #specified in the AlphaGo
                                                                                           #paper.
                
        super().__init__()
        
        self.conv1 = nn.Conv2d(planes_in, planes, kernel_size=kernel_size,
                        stride=stride, padding='same', bias=bias) #Channels go from 256 to 1 to later pass them into
                                                                  #a fully connected layer.
            
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.fc1 = nn.Linear(boardsize**2, 256, bias=bias) #Two fully connected layers with decreasing size  
                                                           #so that the final output has shape batchsize, 1.
            
        self.fc2 = nn.Linear(256, 1 , bias=bias)

    def forward(self, x):
        
        bs = x.shape[0]
        
        out = self.conv1(x)
        
        out= self.bn1(out)
        
        out = F.relu(out)
        
        out= out.view(bs, boardsize**2) #Output of goes from batchsize, 1, 19, 19 to batchsize, 19**2 
        
        out = self.fc1 (out)
        
        out = F.relu(out)
        
        out = self.fc2 (out)
        
        out = torch.tanh(out) #Tanh applied to ouput to scale values from 1 to -1. 
        
        return out

class PolicyHead(nn.Module): #Policy head used to predict what the best move is.
    
    def __init__(self,  planes_in=256, planes=2, kernel_size =1 , stride=1, bias=True):
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(planes_in, planes, kernel_size=kernel_size, #For the Policy head the AlphaGo paper 
                            stride=stride, padding='same', bias=bias)      #specifies downscaling the number of planes 
                                                                           #from 256 to 2 instead of 1.
        
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.fc1 = nn.Linear( 2*boardsize**2, boardsize**2 , bias=bias)

    def forward(self, x):
        
        bs = x.shape[0]
        
        out = self.conv1(x)
        
        out = self.bn1(out)
        
        out = F.relu(out)
        
        out= out.view(bs, 2*boardsize**2) #Output flattened as required to pass as input into the fully connected
                                          #layer.
        out = self.fc1(out)
        
        return out.view(bs, boardsize,boardsize) #Output turned back into a tensor with shape batchsize, 19, 19.

class GoModel(nn.Module): #Model will take input of shape batchsize, 3, 19, 19 and turn it into shape 
                          # batchsize, 256, 19, 19 via convolutions then each value head will give an output of shape
                          # batchsize, 1 and the policy head of shape batchsize, 19, 19.
            
    def __init__(self, in_planes=3, mid_planes=256, blocks = 10):
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3,
                            stride=1, padding='same') #Turning input with three channels into input with 256 channels.
        
        self.bn1 = nn.BatchNorm2d(mid_planes)
        
        self.layers = nn.ModuleList() #Allows for modules to be passed as input into the neural network.
        
        for _ in range(blocks):
            self.layers.append(ResidualBlock()) #Adds a total of ten residual blocks to form the core of the neural
                                                #network (composition).
            
        #Instantiating both heads of the model.   
        self.policy_head = PolicyHead() 
        self.value_head = ValueHead()
        
    def forward(self,x):
        
        x = self.conv1(x)
        
        x = self.bn1(x)
        
        x = F.relu(x)
        
        for block in self.layers: #Loops through all of the ten residual layers.
            x = block(x)
            
        policy = self.policy_head(x)
        
        value = self.value_head(x)
        
        return policy, value
