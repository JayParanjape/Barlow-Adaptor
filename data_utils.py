import random
import os
import sys
import numpy as np

import torch
from torchvision import datasets, models
from torchvision import transforms

from transform_utils import Transform, data_transforms, data_transforms99, data_transforms_office

def get_data_transforms(dataset):
    if dataset=='cataracts':
        data_dir = "/home/ubuntu/Desktop/Domain_Adaptation_Project/data/final_cataracts"
        chosen_data_transform = data_transforms
    elif dataset=='cataracts_trainval':
        data_dir = "/home/ubuntu/Desktop/Domain_Adaptation_Project/data/final_cataracts_trainval"
        chosen_data_transform = data_transforms

    elif dataset=='d99':
        data_dir = "/home/ubuntu/Desktop/Domain_Adaptation_Project/data/final_d99"
        chosen_data_transform = data_transforms99

    elif dataset=='art':
        data_dir = "/home/ubuntu/Desktop/Domain_Adaptation_Project/data/OfficeHomeDataset_10072016/Art"
        chosen_data_transform = data_transforms_office
    elif dataset=='clipart':
        data_dir = '/home/ubuntu/Desktop/Domain_Adaptation_Project/data/OfficeHomeDataset_10072016/Clipart'
        chosen_data_transform = data_transforms_office
    elif dataset=='real_world':
        data_dir = '/home/ubuntu/Desktop/Domain_Adaptation_Project/data/OfficeHomeDataset_10072016/Real World'
        chosen_data_transform = data_transforms_office
    elif dataset=='product':
        data_dir = '/home/ubuntu/Desktop/Domain_Adaptation_Project/data/OfficeHomeDataset_10072016/Product'
        chosen_data_transform = data_transforms_office

    return data_dir, chosen_data_transform

def get_data(dataset, batch_size=32, barlow_batch_size=32, cross_valid_k=1, use_barlow=False):
    data_dir, chosen_data_transform = get_data_transforms(dataset)
    dataset_dict = {}
    dataloaders = {}
    
    if cross_valid_k>1:
        dataloaders_list = []
        dataset_total = datasets.ImageFolder(os.path.join(data_dir,'trainval'),chosen_data_transform['train'])
        class_names = dataset_total.classes
        print("in data utils: ", len(dataset_total))
        train_indices_list, val_indices_list =  crossvalid(len(dataset_total), cross_valid_k)
        for i in range(len(train_indices_list)):
            train_set = torch.utils.data.dataset.Subset(dataset_total,train_indices_list[i])
            val_set = torch.utils.data.dataset.Subset(dataset_total,val_indices_list[i])
            temp_dataloaders = {}
            temp_dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            temp_dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
            dataloaders_list.append(temp_dataloaders)

        return class_names, dataloaders_list

    for x in ['train','val','test']:
        try:
            dataset_dict[x] = datasets.ImageFolder(os.path.join(data_dir,x),chosen_data_transform[x])
        except:
            dataset_dict[x] = datasets.ImageFolder(data_dir,chosen_data_transform[x])

        dataloaders[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=batch_size, shuffle=True, num_workers=4)


    if use_barlow:
        barlow_transform = Transform()
        try:
            barlow_dataset = datasets.ImageFolder(os.path.join(data_dir,'train'), barlow_transform)
        except:
            barlow_dataset = datasets.ImageFolder(data_dir, barlow_transform)

        barlow_dataloader = torch.utils.data.DataLoader(barlow_dataset, batch_size = barlow_batch_size, num_workers=4)
    else:
        barlow_dataloader=None

    class_names = dataset_dict['train'].classes
    print(len(dataloaders['train']))
    return class_names, dataloaders, barlow_dataloader

# define a cross validation function
def crossvalid(total_size=0,k_fold=5):
    train_indices_list = []
    val_indices_list = []
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # msg
#         print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
#               % (trll,trlr,trrl,trrr,vall,valr))
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_indices_list.append(train_indices)
        val_indices_list.append(val_indices)
    print("number of elements in train indices list: ",len(train_indices_list[0]))

    return train_indices_list, val_indices_list

