from data_utils import get_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from utils import *
from data_utils import get_data
from transform_utils import *
from model import *
import sys
from matplotlib import pyplot as plt
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


mode = int(sys.argv[1])
use_sam_modality = True

class_names,d99_dataloaders,_ = get_data("d99", batch_size=128,use_sam=use_sam_modality)
class_names,cata_dataloaders,_ = get_data("cataracts", use_sam=use_sam_modality)


from torchmetrics.classification import MulticlassAUROC, MulticlassROC

import pdb

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class CNNModel(nn.Module):

    def __init__(self, class_names_len, pretrained_weights=''):
        super(CNNModel, self).__init__()
        if pretrained_weights!='':
            self.feature = models.resnet50(pretrained=False)
            self.feature.load_state_dict(torch.load(pretrained_weights),strict=False)
        else:
            # self.feature = models.resnet18(pretrained=True)
            self.feature = models.resnet50(pretrained=True)

        self.in_features = self.feature.fc.in_features
        self.feature.fc = nn.Identity()
        # self.feature = nn.Sequential()
        # self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        # self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        # self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu1', nn.ReLU(True))
        # self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        # self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        # self.feature.add_module('f_drop1', nn.Dropout2d())
        # self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu2', nn.ReLU(True))

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
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, self.in_features)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

#helper functon for testing
def test(model, dataloader, len_classnames, use_dict=False):
    with torch.no_grad():
        my_net = model
        my_net = my_net.eval()
        metric = MulticlassAUROC(num_classes=14, average="macro", thresholds=None)
        metric_wt = MulticlassAUROC(num_classes=14, average="weighted", thresholds=None)
        metric_none = MulticlassAUROC(num_classes=14, average="none", thresholds=None)
        metric_roc = MulticlassROC(num_classes=14, thresholds=None)

        my_net = my_net.to(device)

        my_net = my_net.to(device)
        print("number of classes: ", len_classnames)
        len_dataloader = len(dataloader)
        data_target_iter = iter(dataloader)

        i = 0
        n_total = 0
        n_correct = 0
        total_preds = []
        total_labels = []
        outputs_all = []


        while i < len_dataloader:

            # test model using target data
            data_target = data_target_iter.next()
            t_img, t_label = data_target

            t_img = t_img.to(device)
            t_label = t_label.to(device)

            class_output = my_net(input_data=t_img, alpha=0, mode='test')
            pred = class_output.data.max(1, keepdim=True)[1]
            outputs_all.append(class_output.cpu())


            valid_pred, valid_label = pred, t_label
                
            n_correct += valid_pred.eq(valid_label.data.view_as(valid_pred)).cpu().sum()
            total_preds = total_preds + list(valid_pred.data.cpu().numpy().astype(int).reshape((-1,)))
            total_labels = total_labels + list(valid_label.data.cpu().numpy().astype(int).reshape((-1,)))
            # print(valid_label.size(dim=0))
            n_total += int(valid_label.size(dim=0))

            i += 1

        # print(valid_pred)
        
        accu = n_correct.data.numpy() * 1.0 / n_total
        acc,aacc = visda_acc(total_preds,total_labels)
        # micro_acc = micro_accuracy(total_preds, total_labels)
        print("n_total: ", n_total)
        # print("micro accuracy: ", micro_acc)
        print("visda accuracy: ", acc)
        print("classwise accuracy: ", aacc)
        print(accu)
        cr = (classification_report(total_labels, total_preds))
        print(cr)

        outputs_all = torch.cat(outputs_all,dim=0)
        print("roc auc score: ", metric((outputs_all), torch.tensor(total_labels)))
        print("roc auc score weighted: ", metric_wt((outputs_all), torch.tensor(total_labels)))
        print("roc auc score none: ", metric_none((outputs_all), torch.tensor(total_labels)))

        fpr, tpr, thresholds = metric_roc(outputs_all, torch.tensor(total_labels))
        # plt.plot(fpr[0],tpr[0])

        return fpr, tpr
        # return my_net

def load_model(model_path, class_names, use_vit=True, use_sam_modality=False):
    len_classnames = len(class_names)
    my_net = BARLOW_DANN(len_classnames,lambd=3.9e-3, scale_factor=0.1, use_vit=use_vit, sam_modality=use_sam_modality)
    my_net.load_state_dict(torch.load(model_path))
    my_net = my_net.eval()
    return my_net

def test_model_mycode2(model, test_loader, verbose=True):
    
    with torch.no_grad():
        model.eval()   # Set model to evaluate mode
        model = model.to(device)
        running_corrects = 0
        metric = MulticlassAUROC(num_classes=14, average="macro", thresholds=None)
        metric_wt = MulticlassAUROC(num_classes=14, average="weighted", thresholds=None)
        metric_none = MulticlassAUROC(num_classes=14, average="none", thresholds=None)

        count=0
        preds_all = []
        outputs_all = []
        gold = []
        for inputs, labels in test_loader:
            count+=labels.shape[0]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # pdb.set_trace()
            # print("AUROCmetric: ", metric(outputs, labels))
            preds_all.append(preds.cpu())
            outputs_all.append(outputs.cpu())
            gold.append(labels.data.cpu())
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_acc = running_corrects.double() / count

        gold = np.concatenate(gold)
        preds_all = np.concatenate(preds_all)
        outputs_all = np.concatenate(outputs_all)

        if verbose:
            print(f'Best val Acc: {epoch_acc:4f}')
            matrix = confusion_matrix(gold, preds_all)
            print(matrix)
            classwise_acc = matrix.diagonal()/matrix.sum(axis=1)
            print(classwise_acc)
            print("macro_accuracy: ", np.mean(classwise_acc))
            FP = matrix.sum(axis=0) - np.diag(matrix)  
            FN = matrix.sum(axis=1) - np.diag(matrix)
            TP = np.diag(matrix)
            TN = matrix.sum() - (FP + FN + TP)
            print("TPR: ", (TP/(TP+FN)))
            print("FPR: ", (FP/(FP+TN)))
            print("roc auc score: ", metric(torch.tensor(outputs_all), torch.tensor(gold)))
            print("roc auc score wt: ", metric_wt(torch.tensor(outputs_all), torch.tensor(gold)))
            print("roc auc score none: ", metric_none(torch.tensor(outputs_all), torch.tensor(gold)))

            # print("Macro Recall: ", balanced_accuracy_score(gold,preds_all))
    return preds_all, gold

    # load best model weights

def test_DANN(model_path, dataloader, len_classnames, use_dict=False):
    with torch.no_grad():
        my_net = CNNModel(len_classnames)
        my_net.load_state_dict(torch.load(model_path))
        my_net = my_net.eval()

        my_net = my_net.to(device)

        len_dataloader = len(dataloader)
        data_target_iter = iter(dataloader)

        i = 0
        n_total = 0
        n_correct = 0
        preds_all = []
        outputs_all = []
        gold = []

        while i < len_dataloader:

            # test model using target data
            data_target = data_target_iter.next()
            t_img, t_label = data_target

            batch_size = len(t_label)

            t_img = t_img.to(device)
            t_label = t_label.to(device)

            class_output, _ = my_net(input_data=t_img, alpha=0)
            pred = class_output.data.max(1, keepdim=True)[1]

            preds_all.append(pred.cpu())
            outputs_all.append(class_output.cpu())
            gold.append(t_label.data.cpu())

            i += 1

        gold = np.concatenate(gold)
        preds_all = np.concatenate(preds_all)
        outputs_all = np.concatenate(outputs_all)

        matrix = confusion_matrix(gold, preds_all)
        print(matrix)
        classwise_acc = matrix.diagonal()/matrix.sum(axis=1)
        print(classwise_acc)
        micro = matrix.diagonal().sum()/matrix.sum()
        print("micro accuracy: ",micro)
        print("macro accuracy: ",np.mean(classwise_acc))
        FP = matrix.sum(axis=0) - np.diag(matrix)  
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)
        print("TPR: ", (TP/(TP+FN)))
        print("FPR: ", (FP/(FP+TN)))
        # print("roc auc score: ", metric(torch.tensor(outputs_all), torch.tensor(gold)))
        # print("roc auc score wt: ", metric_wt(torch.tensor(outputs_all), torch.tensor(gold)))
        # print("roc auc score none: ", metric_none(torch.tensor(outputs_all), torch.tensor(gold)))

        return

#target only and source only testing
if mode==1:
    model_r50 = models.resnet50(pretrained=True)
    num_ftrs = model_r50.fc.in_features
    model_r50.fc = nn.Linear(num_ftrs, len(class_names))

    #only cataracts
    # model_r50.load_state_dict(torch.load("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/Transformer_dann_barlow/cataracts_saved_models_2/model_r50_trained_on_final_cataracts_only_ep25-50.pth"))
    #only d99
    model_r50.load_state_dict(torch.load("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/mycode2/model_r50_trained_on_final_d99.pth"))

    model_r50 = model_r50.to(device)

    # test_model_mycode2(model_r50,d99_dataloaders['test'])
    test_model_mycode2(model_r50,cata_dataloaders['test'])

#testing after domain adaptation
elif mode==2:
    #barlow+coral r50
    model_path = "checkpoints/final_checkpoints/src_d99_tgt_cataracts_train_r50_bs16_bm1e-3_loss7_best.pth"
    model = load_model(model_path,class_names,use_vit=False)
    test(model, cata_dataloaders['test'], len(class_names))


    # #barlow+coral timm
    model_path2 = "checkpoints/src_d99_tgt_cataracts_train_timm_bs16_bm1e-4_loss7_effort_2_best.pth"
    model = load_model(model_path2,class_names,use_vit=True)
    test(model, cata_dataloaders['test'], len(class_names))


#testing for only DANN approach
elif mode==3:
    # model_path = "/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/Transformer_dann_barlow/cataracts_saved_models_2/src_d99_tgt_cata_simpledann_best.pth"
    model_path = "/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/Transformer_dann_barlow/cataracts_saved_models_2/src_cata_tgt_d99_simpledann_best.pth"
    test_DANN(model_path, d99_dataloaders['test'], len(class_names))