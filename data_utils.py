import random
import os
import sys
import numpy as np

import torch
from torchvision import datasets, models
from torchvision import transforms

from transform_utils import Transform, data_transforms, data_transforms99, data_transforms_office

#function to choose data transforms and data path based on the dataset name. Change paths here for your own datasets
def get_data_transforms(dataset, data_path=''):
    if dataset=='cataracts':
        if data_path!='':
            data_dir = data_path
        else:
            data_dir = "/home/ubuntu/Desktop/Domain_Adaptation_Project/data/cataracts/cataracts_final"
        chosen_data_transform = data_transforms
    elif dataset=='cataracts_trainval':
        if data_path!='':
            data_dir = data_path
        else:
            data_dir = "/home/ubuntu/Desktop/Domain_Adaptation_Project/data/final_cataracts_trainval"
        chosen_data_transform = data_transforms

    elif dataset=='d99':
        if data_path!='':
            data_dir = data_path
        else:
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

#driver function for creating dataloaders
def get_data(dataset, batch_size=32, cross_valid_k=1, data_path=''):
    data_dir, chosen_data_transform = get_data_transforms(dataset, data_path)
    dataset_dict = {}
    dataloaders = {}
    
    #in case of k fold cross validation, there should be a directory called trainval with all imaegs in the training and validation set in the same folder 
    if cross_valid_k>1:
        dataloaders_list = []
        dataset_total = datasets.ImageFolder(os.path.join(data_dir,'trainval'),chosen_data_transform['train'])
        class_names = dataset_total.classes
        print("length of dataset: ", len(dataset_total))
        train_indices_list, val_indices_list =  crossvalid(len(dataset_total), cross_valid_k)
        for i in range(len(train_indices_list)):
            train_set = torch.utils.data.dataset.Subset(dataset_total,train_indices_list[i])
            val_set = torch.utils.data.dataset.Subset(dataset_total,val_indices_list[i])
            temp_dataloaders = {}
            temp_dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            temp_dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
            dataloaders_list.append(temp_dataloaders)

        return class_names, dataloaders_list

    #in case of predefined train, val and test splits, expected to have separate folders in the root directory
    for x in ['train','val','test']:
        try:
            dataset_dict[x] = datasets.ImageFolder(os.path.join(data_dir,x),chosen_data_transform[x])
        except:
            dataset_dict[x] = datasets.ImageFolder(data_dir,chosen_data_transform[x])

        dataloaders[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=batch_size, shuffle=True, num_workers=4)


    class_names = dataset_dict['train'].classes
    print("length of training data: ",len(dataloaders['train']))
    return class_names, dataloaders

# define a cross validation function
def crossvalid(total_size=0,k_fold=5):
    train_indices_list = []
    val_indices_list = []
    fraction = 1/k_fold
    seg = int(total_size * fraction)

    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_indices_list.append(train_indices)
        val_indices_list.append(val_indices)
    print("number of elements in train indices list: ",len(train_indices_list[0]))

    return train_indices_list, val_indices_list

