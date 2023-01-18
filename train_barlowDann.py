'''
arguments:
1 - source dataset name
2 - target dataset name
3 - use dictionary for source
4 - use dictionary for target
5 - save path
6 - device to use
7 - batch size
8 - barlow batch size
9 - use barlow during training
10 - lr
11 - barlow loss multiplier
12 - DANN multiplier
'''

import torch.optim as optim
from torch.optim import lr_scheduler
import torch
torch.autograd.set_detect_anomaly(True)
import sys

from transformers_dann import *
from data_utils import get_data
from transform_utils import *
from utils import *

#constants and hyperparameters
# device = 'cpu'
device_arg = sys.argv[6]
device = torch.device("cuda:"+device_arg if torch.cuda.is_available() else "cpu")
lr = 8e-4 if sys.argv[10]=='' else float(sys.argv[10])
print(lr)
lr_barlow = 1e-4
batch_sz = int(sys.argv[7])
barlow_batch_sz = int(sys.argv[8])
n_epoch = 100
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

source_dataset = sys.argv[1]
target_dataset = sys.argv[2]
print("detected datasets: ",source_dataset, " ",target_dataset)

source_data_class_names, source_dataloader, source_barlow_dataloader = get_data(source_dataset,batch_size=batch_sz, barlow_batch_size=barlow_batch_sz)
target_data_class_names, target_dataloader, target_barlow_dataloader = get_data(target_dataset,batch_size=batch_sz, barlow_batch_size=barlow_batch_sz)

# load model

my_net = BARLOW_DANN(class_names_len=len(source_data_class_names), lambd=3.9e-3, scale_factor=0.1)
pytorch_total_params = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
print('learnable parameters: ', pytorch_total_params)

use_barlow = True if sys.argv[9].lower()=='true' else False
barlow_loss_multiplier = float(sys.argv[11])
#barlow twins model and loss
if use_barlow:
    learner = BarlowTwins(my_net.feature, -1, [1024,1024, 1024, 1024],
                    3.9e-3, barlow_loss_multiplier)

# setup optimizer

optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.33)
if use_barlow:
    optimizer_barlow = optim.Adam(learner.parameters(), lr = lr_barlow)
    exp_lr_scheduler_barlow = lr_scheduler.StepLR(optimizer_barlow, step_size=40, gamma=0.33)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

my_net = my_net.to(device)
if use_barlow:
    learner = learner.to(device)
loss_class = loss_class.to(device)
loss_domain = loss_domain.to(device)

for p in my_net.parameters():
    p.requires_grad = True

if use_barlow:
    for p in learner.parameters():
        p.requires_grad = True


#helper functon for testing
def test(model_path, dataloader, len_classnames, use_dict=False):
    my_net = BARLOW_DANN(len_classnames,lambd=3.9e-3, scale_factor=0.1)
    my_net.load_state_dict(torch.load(model_path))
    my_net = my_net.eval()

    my_net = my_net.to(device)
    print("number of classes: ", len_classnames)
    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    total_preds = []
    total_labels = []

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        t_img = t_img.to(device)
        t_label = t_label.to(device)

        class_output = my_net(input_data=t_img, alpha=0, mode='test')
        pred = class_output.data.max(1, keepdim=True)[1]

        if use_dict:
            valid_pred, valid_label = mapper_d99_cat(t_label, pred)
        else:
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
    micro_acc = micro_accuracy(total_preds, total_labels)
    print("n_total: ", n_total)
    print("micro accuracy: ", micro_acc)
    print("visda accuracy: ", acc)
    print("classwise accuracy: ", aacc)
    return accu

#load if model available
# my_net.load_state_dict(torch.load('./src_cata_model_best_timmViT.pth'))

# training
save_file = sys.argv[5]
save_path = './saved_models/'+save_file
best_accu_t = 0.0
for epoch in range(n_epoch):

    len_dataloader = min(len(source_dataloader['train']), len(target_dataloader['train']))
    data_source_iter = iter(source_dataloader['train'])
    data_target_iter = iter(target_dataloader['train'])

    if use_barlow:
        data_source_barlow_iter = iter(source_barlow_dataloader)
        data_target_barlow_iter = iter(target_barlow_dataloader)

    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        alpha = float(sys.argv[12])
        # print(alpha)

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source
        data_target = data_target_iter.next()
        t_img, _ = data_target
        min_batch_shape=0

        s_img = torch.unsqueeze(s_img,dim=-1)
        t_img = torch.unsqueeze(t_img, dim=-1)
        try:
            input_data = torch.cat([s_img,t_img], dim=-1).to(device)
        except:
            #for the last iteration there will be a shape mismatch
            min_batch_shape = min(s_img.size(0), t_img.size(0))
            input_data = torch.cat([s_img[:min_batch_shape],t_img[:min_batch_shape]], dim=-1).to(device)
            s_label = s_label[:min_batch_shape]


        my_net.zero_grad()
        #batch size has to be same for source and target
        batch_size = len(s_label) if min_batch_shape==0 else min_batch_shape

        domain_label = torch.zeros(batch_size).long()
        domain_label = domain_label.to(device)
        s_label = s_label.to(device)

        barlow_loss, class_output, src_domain_output, tgt_domain_output = my_net(input_data=input_data, alpha=alpha)
        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(src_domain_output, domain_label)

        batch_size = len(t_img)
        domain_label = torch.ones(batch_size).long()
        domain_label = domain_label.to(device)
        err_t_domain = loss_domain(tgt_domain_output, domain_label)
        
        if use_barlow:
            #barlow twin loss
            source_bt_data,_ = data_source_barlow_iter.next()
            target_bt_data,_ = data_target_barlow_iter.next()
            err_bt_s = learner(source_bt_data[0].to(device), source_bt_data[1].to(device))
            err_bt_t = learner(target_bt_data[0].to(device), target_bt_data[1].to(device))
            err_barlow = err_bt_s + err_bt_t


        err = err_s_label + barlow_loss
        # err = err_t_domain + 0.33*err_s_domain + err_s_label + barlow_loss
        # err = err_s_label
        err.backward()
        if use_barlow:
            err_barlow.backward()

        optimizer.step()
        if use_barlow:
            optimizer_barlow.step()

        if use_barlow:
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f, barlow loss: %f, err_s_bt: %f, err_t_bt: %f' \
            % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item(), barlow_loss.data.cpu().item(),err_bt_s.data.cpu(),err_bt_t.data.cpu()))
        else:
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f,  barlow loss: %f' \
                % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item(), barlow_loss.data.cpu().item()))

        sys.stdout.flush()

        torch.save(my_net.state_dict(), save_path)

    print('\n')
    exp_lr_scheduler.step()
    if use_barlow:
        exp_lr_scheduler_barlow.step()

    use_dict_src = True if (sys.argv[3]).lower()=='true' else False
    accu_s = test(save_path, source_dataloader['val'],len(source_data_class_names),use_dict=use_dict_src)
    print('Accuracy of the %s dataset: %f' % (sys.argv[1], accu_s))
    use_dict_tgt = True if (sys.argv[4]).lower()=='true' else False
    accu_t = test(save_path,target_dataloader['val'],len(source_data_class_names),use_dict=use_dict_tgt)
    print('Accuracy of the %s dataset: %f\n' % (sys.argv[2], accu_t))
    if accu_t > best_accu_t:
        best_accu_s = accu_s
        best_accu_t = accu_t
        torch.save(my_net.state_dict(), f'./src_{sys.argv[1]}_tgt_{sys.argv[2]}_model_best_timmViT.pth')

    print("best accuracy source: ", best_accu_s)
    print("best accuracy target: ", best_accu_t)

print("best accuracy source: ", best_accu_s)
print("best accuracy target: ", best_accu_t)