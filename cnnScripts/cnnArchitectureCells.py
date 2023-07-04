#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:42:16 2023

@author: listonlab
"""
## 0. Import

from Scripts.Data import Data
import pickle
import pandas as pd
import numpy as np
from numpy.random import seed; seed(111)
import random
from scipy.stats import ttest_ind
from IPython.display import Image
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import itertools
from itertools import product

def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)

class ClassificationBase(nn.Module):
    def training_step(self, batch):
        #inputs, classes = batch
        images, targets = batch
        images = images.type(torch.FloatTensor) # Uncomment for BreastCancer ClassfierBase class
        #images = torch.reshape(images.type(torch.DoubleTensor), (len(images), 1))
        targets = torch.reshape(targets.type(torch.FloatTensor), (len(targets), 1))
        out = self(images)
        loss = F.binary_cross_entropy(out, targets)
        return loss

    def validation_step(self, batch):
        images, targets = batch
        images = images.type(torch.FloatTensor) # Uncomment for BreastCancer ClassfierBase class
        #images = torch.reshape(images.type(torch.DoubleTensor), (len(images), 1))
        #print(images)
        targets = torch.reshape(targets.type(torch.FloatTensor), (len(targets), 1))
        #print(targets)
        out = self(images)                           # Generate predictions
        loss = F.binary_cross_entropy(out, targets)  # Calculate loss
        score = F_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() }

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_score']))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda') #REQUIRES CHANGING THE TORCH.FLOATTENSOR TO TORCH.CUDA.FLOATTENSOR
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    

# %load_ext tensorboard

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    #writer = SummaryWriter()  # Create a SummaryWriter instance

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []  # learning rate
        step = 0  # Initialize the step counter
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Write the training loss to TensorBoard with unique step for each batch
            writer.add_scalar('Training Batch Loss', loss, step)
            step += 1  # Increment the step counter

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Write the training loss and learning rate to TensorBoard
        writer.add_scalar('Training Loss', torch.stack(train_losses).mean().item(), epoch)
        writer.add_scalar('Learning Rate', lrs[-1], epoch)

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

    return history

def plot_scores(history):
    scores = [x['val_score'] for x in history]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('F1 score vs. No. of epochs')
    plt.show()
    #plt.savefig("DNN_scores_no_augmentation")

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()
    #plt.savefig("DNN_losses_no_augmentation")

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    plt.show()
    #plt.savefig("DNN_lrs_no_augmentation")
    


device = get_default_device()
device

file = open("C:/repos/MTR/data/train_test1.dat","rb")
train_dataset = pickle.load(file)
file.close()

file = open("C:/repos/MTR/data/test_test1.dat","rb")
val_dataset = pickle.load(file)
file.close()


train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=True)

train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)



class CNNGridSearch(ClassificationBase):
    def __init__(
        self,
        input_channels,
        conv1_filters,
        conv1_kernel_size,
        conv2_filters,
        conv2_kernel_size,
        maxpool_kernel_size,
        dropout,
        fc1_nodes,
        # use_second_dense,
        fc2_nodes,
        # use_third_dense,
        fc3_nodes,
        use_second_conv_block,
        flat_features,
        conv_stride,
    ):
        super().__init__()
        # channels, height, width = input_shape
        ## First convolutional layer
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=conv1_filters, kernel_size=conv1_kernel_size) #(1,A)? - THE NUMBER OF NODES IN THE INPUT VECTOR. OR JUST KERNEL SIZE = 3?
        self.bn1 = torch.nn.BatchNorm2d(conv1_filters)
        self.act1 = nn.ReLU()

        ## Second (optional) convolutional layer
        self.use_second_conv_block = use_second_conv_block
        if self.use_second_conv_block:
            self.conv2 = nn.Conv2d(in_channels = conv1_filters, out_channels=conv2_filters, kernel_size=conv2_kernel_size)
            self.bn2 = torch.nn.BatchNorm2d(conv2_filters)
            self.act2 = nn.ReLU()

            ## Pooling layer
            self.pool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=conv_stride)
            self.flat = nn.Flatten()
        else:

          ## Pooling layer
          self.pool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=conv_stride)
          self.flat = nn.Flatten()

        ## Dense layers
        # "The dense layer further extracts information from the pooling layer."
        self.fc1 = nn.Linear(in_features=flat_features, out_features=fc1_nodes) #flat_features = 10
        self.bn3 = torch.nn.BatchNorm1d(fc1_nodes)
        self.act3 = nn.ReLU()
        self.do1 = nn.Dropout(p=dropout)

        # Second dense layer
        self.fc2 = nn.Linear(in_features=fc1_nodes, out_features=fc2_nodes)
        self.bn4 = torch.nn.BatchNorm1d(fc2_nodes)
        self.act4 = nn.ReLU()
        self.do2 = nn.Dropout(p=dropout)

        # Third dense layer
        self.fc3 = nn.Linear(in_features=fc2_nodes, out_features=fc3_nodes)
        self.bn5 = torch.nn.BatchNorm1d(fc3_nodes)
        self.act5 = nn.ReLU()
        self.do3 = nn.Dropout(p=dropout)

        ## Output layer
        # "Uses logistic regression to report the probability of AD for each sample."
        self.fc5 = nn.Linear(in_features=fc3_nodes, out_features=1)
        self.bn6 = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)
        #print("Input dimensions", x.shape)
        x = self.act1(self.bn1(self.conv1(x)))
        #print("Input dimensions 1st conv layer", x.shape)
        if self.use_second_conv_block:
            x = self.act2(self.bn2(self.conv2(x)))
            x = self.flat(self.pool(x))
            #print("Input dimensions flat features conv2", x.shape)
        else:
            x = self.flat(self.pool(x))
            #print("Input dimensions flat features conv1", x.shape)

        x = self.do1(self.act3(self.bn3(self.fc1(x))))
        x = self.do2(self.act4(self.bn4(self.fc2(x))))
        x = self.do3(self.act5(self.bn5(self.fc3(x))))
        out = self.bn6(self.fc5(x))
        return self.sigmoid(out)

opt_func = torch.optim.Adam


# Define the hyperparameter values to explore
hyperparameters = {
    'epochs': [50, 100],
    'max_lr': [0.01, 0.001],
    'conv1_filters': [1], #,3, 5],
    'conv1_kernel_size': [(2,2)],
    'conv2_filters': [1],#, 3, 5],
    'conv2_kernel_size': [(2,2)],#, 5, 9],
    'conv_stride': [2],
    'maxpool_kernel_size': [(2,2)],
    'dropout': [0.1],
    'fc1_nodes': [2048],
    'fc2_nodes': [512],
    'fc3_nodes': [128],
    'use_second_conv_block': [True, False]
}

# Create a grid of all possible combinations as dictionaries
grid = [dict(zip(hyperparameters.keys(), combination)) for combination in itertools.product(*hyperparameters.values())]

res = {"Parameters":[], "Score":[]}
best_score = 0.0
best_epochs = 0.0
best_lr = 0.0

# Iterate over each combination in grid and access parameter values by name
for par in grid:
    res["Parameters"].append(par)
    e = par['epochs']
    lr = par['max_lr']
    conv1_filter = par['conv1_filters']
    conv1_kernel = par['conv1_kernel_size']
    conv2_filter = par['conv2_filters']
    conv2_kernel = par['conv2_kernel_size']
    stride = par['conv_stride']
    maxpool_kernel_size = par['maxpool_kernel_size']
    dropout = par['dropout']
    fc1 = par['fc1_nodes']
    fc2 = par['fc2_nodes']
    fc3 = par['fc3_nodes']
    use_second= par['use_second_conv_block']
    
  
    # Create a unique tag for each run based on the hyperparameters
    tag = f"{e}_{lr}_{conv1_filter}_{conv1_kernel}_{use_second}"
    print("Running", tag)

    # Create a SummaryWriter instance for each run
    writer = SummaryWriter(log_dir=f"runs/gridCNN0407/{tag}")

    print("Device:", device)
    model = to_device(CNNGridSearch(input_channels=1,
                                    conv1_filters=conv1_filter,
                                    conv1_kernel_size=conv1_kernel,
                                    maxpool_kernel_size=maxpool_kernel_size,
                                    conv2_filters=conv2_filter,
                                    conv2_kernel_size=conv2_kernel,
                                    conv_stride = stride,
                                    use_second_conv_block=use_second,
                                    dropout=dropout,
                                    fc1_nodes=fc1,
                                    fc2_nodes=fc2,
                                    fc3_nodes=fc3,
                                    # Hardcoded FIX
                                    flat_features=9801), device)
    
    # Train the model and evaluate its performance
    history = [evaluate(model, val_dl)]
    history += fit_one_cycle(e, lr, model, train_dl, val_dl, opt_func=opt_func)
    
    # Calculate the validation score
    final_score = history[-1]['val_score']
    scores["Score"].append(final_score)
    # Check if the current combination is the best
    if final_score > best_score:
        print("Better parameters found, updating.")
        best_score = final_score
        best_epochs = e
        best_max_lr = lr
        
    # Close the writer for each run
    writer.close()
    
# Print the best hyperparameters
print("Best Hyperparameters:")
print("Epochs:", best_epochs)
print("Max LR:", best_max_lr)

