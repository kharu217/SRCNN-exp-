#%%

#import necessary library
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import Dataset, DataLoader

import cv2
import os
import numpy as np
import glob
import random
import matplotlib.pyplot as plt

from model import SRCNN, train_dataset

#out_channel numbers
n1, n2, n3 = 128, 64, 3

#filters(kernels) size
f1, f2, f3 = 9, 3, 5

upscale_factor = 5

input_size = 33
output_size = input_size - f1 - f2 - f3 + 3

stride = 14

#train hyperparam
batch_size = 500
epochs = 20

min_loss = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = r"C:\Users\chris\OneDrive\바탕 화면\Git_Hub\SRCNN\Data_SRCNN\T91"
save_path = r"models\normal.h5"


#train
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

print(len(train_dataloader.dataset))

model_t = SRCNN().to(device)
params = model_t.parameters()

optimizer = optim.Adam(params=params, lr=1e-3)
loss_fn = nn.MSELoss()

for i in range(epochs) :
    print("{} Epochs...".format(i + 1))
    model_t.train()

    size = len(train_dataloader.dataset)
    
    for batch, (X, y) in enumerate(train_dataloader) :
        X = X.to(device)
        y = y.to(device)
        
        pred = model_t(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad() #변화도 0으로 초기화
        loss.backward() #변화도에 각각의 파라미터의 손실 변화도 +
        optimizer.step() #현재 변화도를 토대로 파라미터 조정
        
        if batch % 100 == 0 :
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")
            if loss < min_loss :
                min_loss = loss
                print('new_loss_best')
                torch.save(model_t.state_dict(), save_path)
        if loss < min_loss :
                min_loss = loss
                torch.save(model_t.state_dict(), save_path)
print(min_loss)
    
print("Done")

# %%
