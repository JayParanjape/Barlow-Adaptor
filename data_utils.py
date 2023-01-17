import random
import os
import sys
import numpy as np

import torch
from torchvision import datasets, models
from torchvision import transforms

from transform_utils import Transform, data_transforms, data_transforms99

def get_data(dataset, batch_size=32, barlow_batch_size=32):
    if dataset=='cataracts':
        data_dir = "../../data/cataracts/instrument_split_15/"
        
    elif dataset=='d99':
        data_dir = "../../data/cataracts/dataset99_ins_split_balanced_phoenix/image_extracts_balanced/"
    
    elif dataset=='art':
        data_dir = "/home/ubuntu/Desktop/Domain_Adaptation_Project/data/OfficeHomeDataset_10072016/Art"
    elif dataset=='clipart':
        data_dir = '/home/ubuntu/Desktop/Domain_Adaptation_Project/data/OfficeHomeDataset_10072016/Clipart'
    elif dataset=='real_world':
        data_dir = '/home/ubuntu/Desktop/Domain_Adaptation_Project/data/OfficeHomeDataset_10072016/Real World'
    elif dataset=='product':
        data_dir = '/home/ubuntu/Desktop/Domain_Adaptation_Project/data/OfficeHomeDataset_10072016/Product'

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    dataset_dict = {}
    dataloaders = {}
    
    for x in ['train','val','test']:
        try:
            dataset_dict[x] = datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
        except:
            dataset_dict[x] = datasets.ImageFolder(data_dir,data_transforms[x])

        dataloaders[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=batch_size, shuffle=True, num_workers=4)


    barlow_transform = Transform()
    try:
        barlow_dataset = datasets.ImageFolder(os.path.join(data_dir,'train'), barlow_transform)
    except:
        barlow_dataset = datasets.ImageFolder(data_dir, barlow_transform)

    barlow_dataloader = torch.utils.data.DataLoader(barlow_dataset, batch_size = barlow_batch_size, num_workers=4)

    class_names = dataset_dict['train'].classes
    print(len(dataloaders['train']))
    return class_names, dataloaders, barlow_dataloader
