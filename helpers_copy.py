import os
import pandas as pd
import torch
from torch.utils.data import Dataset

## a custom Dataset class that reads in a specifically-formatted pkl file of power usage data ##

# this class converts this data into Tensors that can be batch processed 
class PowerConsumptionDataset(Dataset):
    """Household power consumption dataset class."""

    def __init__(self, pkl_file, transform=None):
        """
        :param pkl_file (string): Path to a binary pickle file.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        power_usage_frame = pd.read_pickle(pkl_file)

        # assumes all columns are input features except the last, which is the target
        input_features = power_usage_frame.iloc[:, :-1].values.astype(dtype = 'float32')
        target = power_usage_frame.iloc[:, -1:].values.astype(dtype = 'float32')
        
        self.x = torch.tensor(input_features, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32)
        
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.transform:
            self.x = self.transform(self.x)
            self.y = self.transform(self.y)

        return self.x[idx], self.y[idx]

    
## training scripts that train and save a model ##

MODEL_DIR = 'saved_models/'

# helper function to save a model
def save_model(model, model_name, model_dir=MODEL_DIR):
    print("Saving the model as " + model_name)
    path = os.path.join(model_dir, model_name)
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)

# train loop
def train(model, train_loader, epochs, optimizer, criterion, 
          model_name = 'model.pth', model_dir = MODEL_DIR, device='cpu'):
    """
    Training loop which returns a trained model (and saves it).
    
    :param model: the PyTorch model that we wish to train.
    :param train_loader: the DataLoader used for training.
    :param epochs: Total number of times to iterate through the training data.
    :param optimizer: optimizer to use during training.
    :param criterion: loss function used for training. 
    :param device: where the model and data should be loaded (gpu or cpu).
    :return: trained model.
    """
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            # prep data
            inputs, target = data.to(device), target.to(device)
            
            optimizer.zero_grad() # zero accumulated gradients
            
            # get output of SimpleNet
            output = model(inputs)
            
            # calculate loss and perform backprop
            # using sqrt to get RMSE rather than just MSE
            loss = torch.sqrt(criterion(output, target))
            
            # backprop + update step
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        
        # print loss stats
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

    # save and return trained model, after all epochs
    save_model(model, model_name, model_dir)
    return model


## test script that evaluates the RMSE of a model on a test set##

def test_eval(model, test_loader, criterion):

    # initialize test loss
    test_loss = 0.0

    model.eval() # prep model for evaluation 

    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss - RMSE
        loss = torch.sqrt(criterion(output, target))
        # update test loss 
        test_loss += loss.item()*data.size(0)

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.sampler)
    
    return test_loss
