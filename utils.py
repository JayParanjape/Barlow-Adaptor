import numpy as np
import torch

def mapper_d99_cat(labels_d99, preds_d99):
    labels_99_arr = labels_d99.cpu().numpy()
    preds_99_arr = preds_d99.cpu().numpy()
    #the keys of this are the labels in the dataset 99 and the values are the corresponding labels in the cataracts dataset
    mapper_dict = {
        0:12,
        1:0,
        2:1,
        3:11,
        4:3,
        5:2,
        6:5,
        7:10,
        8:6,
        9:8,
        10:4,
        11:-1,
        12:9,
        13:-1,
        14:-1,
        15:-1,
        16:-1
    } 

    converted_labels = np.array([mapper_dict[l] for l in labels_99_arr])
    valid_idxs = np.where(converted_labels!=-1)[0]
    valid_labels = converted_labels[valid_idxs]
    valid_preds = preds_99_arr[valid_idxs]
    return torch.Tensor(valid_preds), torch.Tensor(valid_labels)