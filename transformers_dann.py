import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np

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

import torch
import torch.nn as nn
'''
Implementation of Barlow Twins (https://arxiv.org/abs/2103.03230), adapted for ease of use for experiments from
https://github.com/facebookresearch/barlowtwins, with some modifications using code from 
https://github.com/lucidrains/byol-pytorch
'''

def flatten(t):
    return t.reshape(t.shape[0], -1)

class NetWrapper(nn.Module):
    # from https://github.com/lucidrains/byol-pytorch
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)

        return representation



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    '''
    Adapted from https://github.com/facebookresearch/barlowtwins for arbitrary backbones, and arbitrary choice of which
    latent representation to use. Designed for models which can fit on a single GPU (though training can be parallelized
    across multiple as with any other model). Support for larger models can be done easily for individual use cases by
    by following PyTorch's model parallelism best practices.
    '''
    def __init__(self, backbone, latent_id, projection_sizes, lambd, scale_factor=1):
        '''

        :param backbone: Model backbone
        :param latent_id: name (or index) of the layer to be fed to the projection MLP
        :param projection_sizes: size of the hidden layers in the projection MLP
        :param lambd: tradeoff function
        :param scale_factor: Factor to scale loss by, default is 1
        '''
        super().__init__()
        self.backbone = backbone
        self.backbone = NetWrapper(self.backbone, latent_id)
        self.lambd = lambd
        self.scale_factor = scale_factor
        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2, use_projector=True):
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        if use_projector:
            z1 = self.projector(z1)
            z2 = self.projector(z2)

        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c.div_(z1.shape[0])


        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor*(on_diag + self.lambd * off_diag)
        return loss


# if __name__=='__main__':
#     import torchvision


#     model = torchvision.models.resnet18(zero_init_residual=True)
#     proj = [512, 512, 512, 512]
#     twins = BarlowTwins(model, 'avgpool', proj, 0.5)
#     inp1 = torch.rand(2,3,224,224)
#     inp2 = torch.rand(2,3,224,224)
#     outs =twins(inp1, inp2)
#     #model = model_utils.extract_latent.LatentHook(model, ['avgpool'])
#     #out, dicti = model(inp1)
#     print(outs)
#     #print(model)

class ViTModels(nn.Module):
    def __init__(self, image_size = 224, in_features=1024, mode='timm'):
        super().__init__()
        self.in_features = in_features
        if mode=='Simple':
            V = SimpleViT(
                image_size = image_size,
                patch_size = 16,
                num_classes = in_features,
                dim = 1024,
                depth = 2,
                heads = 16,
                mlp_dim = 2048
            )
            self.vitModel = V
        else:
            timm_model = timm.create_model('vit_base_patch16_224', num_classes=in_features, pretrained=True)
            timm.models.load_checkpoint(timm_model, "vit_checkpoints/ViT-B_16-224.npz")
            print("Loaded Checkpoint of ViT vit_checkpoints/ViT-B_16-224.npz")
            self.vitModel = timm_model
    def forward(self, x):
        return self.vitModel(x)


class BARLOW_DANN(nn.Module):

    def __init__(self, class_names_len, lambd, scale_factor=1, pretrained_weights=''):
        super(BARLOW_DANN, self).__init__()
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

        self.domain_softmax = nn.Sequential()
        self.domain_softmax.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_softmax.add_module('d_softmax', nn.LogSoftmax(dim=1))
        self.lambd = lambd
        self.scale_factor = scale_factor


    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, self.feature.in_features)
        class_output = self.class_classifier(feature)
        # class_output = feature
        domain_output = self.domain_classifier(feature)
        domain_logits = self.domain_softmax(domain_output)

        # empirical cross-correlation matrix
        c = torch.mm(domain_output.T, domain_output)
        c.div_(domain_output.shape[0])


        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        barlow_loss = self.scale_factor*(on_diag + self.lambd * off_diag)
        return barlow_loss, class_output, domain_logits
