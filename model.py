import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import datasets, models
from torchvision import transforms

from vit_pytorch import SimpleViT
import timm
from transform_utils import Transform



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANN(nn.Module):

    def __init__(self, class_names_len, pretrained_weights=''):
        super(DANN, self).__init__()
        if pretrained_weights!='':
            self.feature = ViTModels(in_features=class_names_len)
            self.feature.load_state_dict(torch.load(pretrained_weights))
        else:
            self.feature = ViTModels(in_features=1024)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.feature.in_features, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, class_names_len))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.feature.in_features, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, self.feature.in_features)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        # class_output = feature
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


'''
Implementation of Barlow Twins (https://arxiv.org/abs/2103.03230), adapted for ease of use for experiments from
https://github.com/facebookresearch/barlowtwins, with some modifications using code from 
https://github.com/lucidrains/byol-pytorch
'''

def flatten(t):
    return t.reshape(t.shape[0], -1)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class ViTModels(nn.Module):
    def __init__(self, image_size = 224, in_features=1024, mode='timm'):
        super().__init__()
        self.in_features = in_features        
        timm_model = timm.create_model('vit_base_patch16_224', num_classes=in_features, pretrained=True)
        timm.models.load_checkpoint(timm_model, "vit_checkpoints/ViT-B_16-224.npz")
        print("Loaded Checkpoint of ViT vit_checkpoints/ViT-B_16-224.npz")

        self.vitModel = timm_model
    def forward(self, x):
        return self.vitModel(x)


class BARLOW_DANN(nn.Module):

    def __init__(self, class_names_len, lambd, scale_factor=1, pretrained_weights='', use_vit=True):
        super(BARLOW_DANN, self).__init__()
        if pretrained_weights!='':
            if use_vit:
                self.feature = ViTModels(in_features=class_names_len)
                self.in_features = self.feature.in_features
            else:
                self.feature = models.resnet50()
                self.in_features = self.fc.in_features
                self.feature.fc = nn.Identity()

            self.feature.load_state_dict(torch.load(pretrained_weights),strict=False)
        else:
            if use_vit:
                self.feature = ViTModels(in_features=1024)
                self.in_features = self.feature.in_features
            else:
                self.feature = models.resnet50(pretrained=True)
                self.in_features = self.feature.fc.in_features
                self.feature.fc = nn.Identity()


        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.in_features, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, class_names_len))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.in_features, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))

        self.domain_softmax = nn.Sequential()
        self.domain_softmax.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_softmax.add_module('d_softmax', nn.LogSoftmax(dim=1))
        self.lambd = lambd
        self.scale_factor = scale_factor


    def forward(self, input_data, alpha, mode='train',use_barlow=True, use_coral=False, use_mmd=False):
        if mode=='train':
            # input_data is of the shape batchX3X224X224X2
            src_input = input_data[:,:,:,:,0]
            tgt_input = input_data[:,:,:,:,1]
            src_feature = self.feature(src_input)
            tgt_feature = self.feature(tgt_input)

            src_feature = src_feature.view(-1, self.in_features)
            tgt_feature = tgt_feature.view(-1, self.in_features)

        else:
            # input_data is of the shape batchX3X224X224X2
            src_input = input_data
            src_feature = self.feature(src_input)
            src_feature = src_feature.view(-1, self.in_features)

            class_output = self.class_classifier(src_feature)
            return class_output


        class_output = self.class_classifier(src_feature)
        # class_output = feature
        src_domain_output = self.domain_classifier(src_feature)
        tgt_domain_output = self.domain_classifier(src_feature)

        src_domain_logits = self.domain_softmax(src_domain_output)
        tgt_domain_logits = self.domain_softmax(tgt_domain_output)

        if use_barlow:
            # empirical cross-correlation matrix
            c = torch.mm(src_domain_output.T, tgt_domain_output)
            c.div_(src_domain_output.shape[0])


            # use --scale-loss to multiply the loss by a constant factor
            # see the Issues section of the readme
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            barlow_loss = self.scale_factor*(on_diag + self.lambd * off_diag)
        else:
            barlow_loss=0

        # print("****Use Coral: ", use_coral)
        if use_coral:
            coral_loss = self.scale_factor*CORAL(src_feature, tgt_feature)
        else:
            coral_loss = 0

        if use_mmd:
            mmd = MMD_loss()
            mmd_loss = self.scale_factor*mmd(src_feature, tgt_feature)
        else:
            mmd_loss = 0

        feature_alignment_loss = barlow_loss + coral_loss + mmd_loss

        return feature_alignment_loss, class_output, src_domain_logits, tgt_domain_logits


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    # print("*****xc: ",xc)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    # print("*****xtc: ",xct)


    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    # loss = loss/(4*d*d)

    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss