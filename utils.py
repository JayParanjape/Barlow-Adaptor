import numpy as np
import torch
from sklearn.metrics import confusion_matrix

'''
01 - secondary incision knife - paracenthesis blade
02 - bonn forceps - 0.12 forceps
03 - charleux cannula - AC Cannula
04 - primary incision knife - keratome
05 - capsulorhexis forceps - utrata forceps
07 - capsulorhexis cystotome - cystotome
08 - hydrodissection cannula - hydrodissection cannula
09 - phacoemulsifier handpiece - phaco
11 - implant injector - IOL injector
13 - suture needle - suture
14 - cotton - eweckell sponge
16 - needle holder - needle driver
22 - irrigation - irrigation
25 - micromanipulator - chopper
'''


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

def mapper_cat_d99(labels_d99, preds_d99):
    labels_99_arr = labels_d99.cpu().numpy()
    preds_99_arr = preds_d99.cpu().numpy()
    #the keys of this are the labels in the dataset 99 and the values are the corresponding labels in the cataracts dataset
    mapper_dict = {
        0:1,
        1:2,
        2:5,
        3:4,
        4:10,
        5:6,
        6:8,
        7:-1,
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



def visda_acc(predict, all_label):
    matrix = confusion_matrix(all_label, predict)
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc

def micro_accuracy(predict, all_label):
    predict_np = np.array(predict)
    all_label_np = np.array(all_label)
    return (predict_np==all_label_np).mean()