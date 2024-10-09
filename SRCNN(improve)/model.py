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

#out_channel numbers
n1, n2, n3 = 128, 64, 3

#filters(kernels) size
f1, f2, f3 = 9, 3, 5

upscale_factor = 3

input_size = 33
output_size = input_size - f1 - f2 - f3 + 3

stride = 14

#train hyperparam
batch_size = 128
epochs = 50

min_loss = 999999999

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = r"Normal_img"
save_path = r"models\normal.h5"


class CustomDataset(Dataset) :
    def __init__(self, img_paths, input_size, output_size, stride = 14, upscale_factor = 3) :
        super(CustomDataset, self).__init__()

        self.img_paths = glob.glob(img_paths + '\\*.jpg')
        self.stride = stride
        self.upscale_factor = upscale_factor
        self.sub_lr_imgs = []
        self.sub_hr_imgs = []
        self.input_size = input_size
        self.output_size = output_size
        self.pad = abs(self.input_size - self.output_size) // 2

        print("Start {} Images Pre-Processing".format(len(self.img_paths)))
        for img_path in self.img_paths :
            #print(img_path)
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        
            #mod crop
            h = img.shape[0] - np.mod(img.shape[0], self.upscale_factor)
            w = img.shape[1] - np.mod(img.shape[1], self.upscale_factor)
            
            img = img[:h, :w, :]
            
            #zoom img
            label = img.astype(np.float32)/255.0
            temp_input = cv2.resize(label, dsize=(0, 0), fx = 1/self.upscale_factor, fy = 1/self.upscale_factor, interpolation = cv2.INTER_AREA)
            input = cv2.resize(temp_input, dsize = (0, 0), fx = self.upscale_factor, fy = self.upscale_factor, interpolation = cv2.INTER_CUBIC)

            #Crop : img to sub_imgs
            for h in range(0, input.shape[0] - self.input_size + 1, self.stride) :
                for w in range(0, input.shape[1] - self.input_size + 1, self.stride) :
                    sub_lr_img = input[h:h+self.input_size, w:w+self.input_size, :]
                    sub_hr_img = label[h+self.pad:h+self.pad+self.output_size, w+self.pad:w+self.pad+self.output_size, :]
                    
                    sub_lr_img = sub_lr_img.transpose((2, 0, 1))
                    sub_hr_img = sub_hr_img.transpose((2, 0, 1))
                    
                    self.sub_lr_imgs.append(sub_lr_img)
                    self.sub_hr_imgs.append(sub_hr_img)
                
        print("Finish, Created {} Sub-Images".format(len(self.sub_lr_imgs)))     
        self.sub_lr_imgs = np.array(self.sub_lr_imgs)
        self.sub_hr_imgs = np.array(self.sub_hr_imgs)
        #print(len(self.sub_hr_imgs))
        #print(len(self.sub_lr_imgs))

    def __len__(self) :
        return len(self.sub_lr_imgs)
    
    def __getitem__(self, idx) :
        lr_img = self.sub_lr_imgs[idx]
        hr_img = self.sub_hr_imgs[idx]
        return lr_img, hr_img    

train_dataset = CustomDataset(path, input_size, output_size)
print(len(train_dataset))
#img = cv2.imread(train_dataset.img_paths[12])
# print(img.shape)
# plt.imshow(img)

class SRCNN(nn.Module) :
    def __init__(self, kernel_list=[f1, f2, f3], filters_list=[n1, n2, n3], num_channels = 3) :
        super(SRCNN,self).__init__()
        
        f1, f2, f3 = kernel_list
        n1, n2, n3 = filters_list
        
        self.conv1 = nn.Conv2d(num_channels, n1, kernel_size=f1)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2)
        self.conv3 = nn.Conv2d(n2, n3, kernel_size=f3)
        self.relu = nn.ReLU(inplace=True)
        
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
    
    def forward(self, x) :
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
        
def test(dataloder, model, loss_fn) :
    size = len(dataloder.dataset)
    num_batches = len(dataloder)
    test_loss = 0
    
    with torch.no_grad() :
        for batch, (X, y) in enumerate(dataloder) :
            X = X.to(device)
            y = y.to(device)
        
        pred = model(X)
        test_loss += loss_fn(pred, y)
    test_loss /= num_batches
    print(f"Avg loss : {test_loss:>8d}\n")

# %%