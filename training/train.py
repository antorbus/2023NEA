import torch.nn as nn
import torch
from dataset import GoDataset
from tqdm import tqdm #Displays a progress bar for loops. 
from torch.utils.data import DataLoader 


boardsize =19 
def create_dataloader_train(paths): #In the training loop both dataset and dataloader objects are dynamically
                                    #generated since each time they are instantiated the data is diffrent.
        
    train_len = int(len(paths)*0.9)
    tenk_dataset_train = GoDataset(paths[:train_len], False) #Setting the test flag to false.
    
    train_dl = DataLoader(tenk_dataset_train, 400, shuffle=True, num_workers=16, drop_last=True)
    #In the dataloader object drop_last is set to true to avoid
    #having problems due to diffrent batchsizes, since shuffle is on this won't have an impact.
    
    return train_dl



def create_dataloader_test(paths):
    train_len = int(len(paths)*0.9)
    tenk_dataset_test = GoDataset(paths[train_len:])
    test_dl = DataLoader(tenk_dataset_test, 50, shuffle=False, num_workers=16)
    return test_dl




#Main training loop that carries out model training and validation. 
def train(model_version, model, epochs, optimizer, scheduler, paths, device, checkpoint=None):
    
    if checkpoint: #If a checkpoint is passed then the model is loaded with the checkpoint to resume training from
                   #that point.
            
        model.load_state_dict(torch.load(checkpoint))
        
    model = model.to(device) #Performs device conversion and transfers model from main memory to GPU memory. 
    
    #Loss function for the value and policy head respectively. 
    criterion_policy = nn.CrossEntropyLoss() 
    criterion_value = nn.MSELoss()
    
    test_dl = create_dataloader_test(paths) 
    
    for epoch in range(1,epochs+1): #The number of epochs is the number of passes through the whole dataset.
        
        train_dl = create_dataloader_train(paths) #Training dataloader objects created dynamically.
        
        train_loss=0
        test_loss=0
        correct_policy = 0
        correct_value = 0
        total = 0
        
        train_dl_pbar = tqdm(train_dl)
        
        model.train() #Model set to training mode.
        for idx, (x, gt, winner) in enumerate(train_dl_pbar):
            
            #Input and output pairs are transferred to the GPU and set to float because the model's weights are also
            #float.
            x, gt, winner = x.to(device).float(), gt.to(device).float(), winner.to(device).float()
            
            #Sets all gradients of the network to zero so that they are recalculated with each batch.
            optimizer.zero_grad()
            
            output = model(x) #Input passed into the model.
            
            #Calculating the policy and value loss of the neural network.
            loss_policy =  criterion_policy(output[0], gt) 
            loss_value =  criterion_value(output[1].squeeze(-1), winner) 
            
            loss = loss_policy + loss_value 
            
            loss.backward() #Calling the backward method to calculate the gradients of the loss with respect to the
                            #models weights on each batch.
            
            optimizer.step() #Optimizing the models weights (learning).
            
            scheduler.step() #Updating the learning rate scheduler. 
            
            #TQDM allows for progress bars to show information, this is useful to check the loss and learning rate
            #mid epoch.
            train_dl_pbar.set_description(f'Loss: {loss:.4f} Lr: {scheduler.get_last_lr()[0]:.6f}')
            
            train_loss += loss.item()*x.size(0) 
            
            
            
        train_loss = train_loss/len(train_dl.dataset) #Average loss over the training cycle.
        
        with torch.no_grad(): #Disables gradient so that model evaluation occupies less space in GPU memory.
            
            model.eval() #Model set to evaluation mode.
            
            test_dl_pbar = tqdm(test_dl)
            
            for (x, gt, winner) in test_dl_pbar:
                
                bs = x.shape[0]
                
                x, gt, winner = x.to(device).float() ,gt.to(device).float(), winner.to(device).float()
                
                output = model(x)
                
                loss_policy =  criterion_policy(output[0], gt)
                
                loss_value = criterion_value(output[1].squeeze(-1), winner)
                
                loss = loss_policy+loss_value
                
                test_dl_pbar.set_description(f'Loss: {loss:.4f}')
                
                test_loss += loss.item()*x.size(0)
                
                pred = output[0].view(bs,-1).data.max(-1, keepdim=True)[1].squeeze(-1)
                
                #Predictions compared against correct output pair and evaluated if match precisely. 
                correct_policy +=(gt.view(bs,-1).data.max(-1, keepdim=True)[1].squeeze(-1)==pred).sum()
                correct_value += (torch.round(output[1]).squeeze(-1).int() == winner.int()).sum()
                
                total += pred.shape[0]  
                
        test_loss = test_loss/len(test_dl.dataset)
        
        #Model saved on each epoch.
        torch.save(model.state_dict(), f'models/m{model_version}-{epoch}.ckpt')
        
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTest Loss {test_loss:.6f}\tTest Accuracy (policy & value): {100*correct_policy/total:.2f}% {100*correct_value/total:.2f}%')
