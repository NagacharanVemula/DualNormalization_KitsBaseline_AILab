import os
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from model.unetdsbn import Unet2D
from datasets.dataset import Dataset, ToTensor, CreateOnehotLabel
# from test_utils import get_bn_statis, cal_distance
import torch
import torchvision.transforms as tfs
from torch import optim
from torch.optim import Adam
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import logging
import copy
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle as pkl
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    SplitDimd,
    LabelToMaskd,
    Rotate90d,
)
import nibabel as nib
from monai.data import Dataset, NibabelReader
from monai.data import DataLoader, decollate_batch
from monai.inferers import SliceInferer
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.nn import PairwiseDistance



data_interest= ""
with open(data_interest,'rb') as f: 
    tr, val, ts = pkl.load(f)



#Utils
def get_bn_statis(model, domain_id):
    means = []
    vars = []
    for name, param in model.state_dict().items():
        if 'bns.{}.running_mean'.format(domain_id) in name:
            means.append(param.clone())
        elif 'bns.{}.running_var'.format(domain_id) in name:
            vars.append(param.clone())
    return means, vars


def cal_distance(means_1, means_2, vars_1, vars_2):
    pdist = PairwiseDistance(p=2)
    dis = 0
    for (mean_1, mean_2, var_1, var_2) in zip(means_1, means_2, vars_1, vars_2):
        dis += (pdist(mean_1.reshape(1, mean_1.shape[0]), mean_2.reshape(1, mean_2.shape[0])) + pdist(var_1.reshape(1, var_1.shape[0]), var_2.reshape(1, var_2.shape[0])))
    return dis.item()




#transforms
test_transforms = Compose([

    LoadImaged(
        keys=["image", "label"], 
        reader=NibabelReader, 
        image_only=False
    ),
    

    EnsureChannelFirstd(
        keys=["image", "label"]
    ),

    LabelToMaskd(
        keys="label",
        select_labels=[1],
        merge_channels=False
    ),
    
      ScaleIntensityRanged(
        keys="image",
        a_min=-100,   
        a_max=300,    
        b_min=-1,      
        b_max=1,       
        clip=True      
    ),
    
    ])
    

#Creating the data loader
test_dataset = Dataset(ts, transform = test_transforms)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle = False, num_workers=4)


#load the trained model

model = Unet2D(num_classes=2, norm='dsbn', num_domains=2)
device = torch.device("cuda:2")
model.load_state_dict(torch.load("best_model_DN.pth")["model_dict"])

model.to(device)


#retrieving the statistics of the trained model
means_list = []
vars_list = []

for i in range(2):
    means, vars = get_bn_statis(model, i)
    means_list.append(means)
    vars_list.append(vars)



#Custom predictor wrappers for Sliding window inference
    
def predictor_wrapper1(data):

    data = data.squeeze(-1) 

    domain_label = domain_id * torch.ones(data.shape[0], dtype=torch.long, device=data.device)
    
    output = model_copy(data, domain_label=domain_label).unsqueeze(-1)
   
    return output


def predictor_wrapper2(data):

    data = data.squeeze(-1) 
    domain_label = selected_domain * torch.ones( data.shape[0], dtype=torch.long, device=data.device)

    output = model(data, domain_label=domain_label).unsqueeze(-1)
    
    
    return output




#Dice metric
dice_metric = DiceMetric(include_background=False, reduction="mean")




#Inference using style based path selection
model_copy = copy.deepcopy(model)
model_copy.train()   

with torch.no_grad():
    for idx, (batch) in enumerate(test_loader):
        model.train()      
    
        sample_data = batch['image'] #Modification: sliding window handles sending things to gpu

        mask = batch['label']
        
        dis = 99999999
        best_out = None
        for domain_id in range(2):

            roi_size = (512, 512,1)

            #Forward pass to get the upated statistics
            output = sliding_window_inference(
                inputs=sample_data,  
                roi_size=roi_size,       
                sw_batch_size= sample_data.shape[0],          #batch size
                predictor=predictor_wrapper1, 
                sw_device="cuda:2", 
                device="cpu" 
            )

            #calculating the updated statistics
            means, vars = get_bn_statis(model_copy, domain_id)

            #calculating the distance
            new_dis = cal_distance(means, means_list[domain_id], vars, vars_list[domain_id])
            
            #selcting the best domain
            if new_dis < dis:
                selected_domain = domain_id
                dis = new_dis

        model_copy = copy.deepcopy(model)
        #Inference using the selcted domain
        model.eval()
        roi_size = (512, 512,1)
        
        output_selected = sliding_window_inference(
                inputs=sample_data,  
                roi_size=roi_size,      
                sw_batch_size= 1,
                predictor=predictor_wrapper2, 
                sw_device="cuda:2", 
                device="cpu" 
            )
        
    
        predicted_mask = torch.argmax(output_selected, dim=1).unsqueeze(1)
        
        dice_metric(y_pred=predicted_mask, y=mask)
        print(dice_metric.get_buffer())
    
    metric = dice_metric.aggregate().item()

    print("mean dice:",metric)
    