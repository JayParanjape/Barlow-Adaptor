'''
here we first pretrain the model with barlow twins self supervised method. then we use the model ths trained for DANN
'''
import torch.optim as optim

from transformers_dann import *
from data_utils import *
from transform_utils import *
from utils import *

#constants and hyperparameters
# device = 'cpu'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
lr = 1e-3
lr_barlow = 1e-3
batch_size = 128
image_size = 28
n_epoch = 100
n_epoch_barlow = 100
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load model

my_net = DANN(class_names_len=len(cataracts_class_names))
#barlow twins model and loss
learner = BarlowTwins(my_net.feature, -1, [512,1024, 1024, 1024],
                3.9e-3, 1)

# setup optimizer

optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=0.9)
optimizer_barlow = optim.Adam(learner.parameters(), lr = lr_barlow)
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

my_net = my_net.to(device)
learner = learner.to(device)
loss_class = loss_class.to(device)
loss_domain = loss_domain.to(device)

for p in my_net.parameters():
    p.requires_grad = True

for p in learner.parameters():
    p.requires_grad = True

#setup source and target datasets
dataloader_source = cataracts_dataloaders
dataloader_target = d99_dataloaders

#helper functon for testing
def test(model_path, dataloader, len_classnames, use_dict=False):
    my_net = DANN(len_classnames)
    my_net.load_state_dict(torch.load(model_path))
    my_net = my_net.eval()

    my_net = my_net.to(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        t_img = t_img.to(device)
        t_label = t_label.to(device)

        class_output, _ = my_net(input_data=t_img, alpha=0)
        pred = class_output.data.max(1, keepdim=True)[1]

        if use_dict:
            valid_pred, valid_label = mapper_d99_cat(t_label, pred)
        else:
            valid_pred, valid_label = t_label, pred
            
        n_correct += valid_pred.eq(valid_label.data.view_as(valid_pred)).cpu().sum()
        # print(valid_label.size(dim=0))
        n_total += int(valid_label.size(dim=0))

        i += 1

    print(valid_pred)

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu

save_path = './saved_models/src_cata_tgt_d99_model_epoch_current_train_pretrain_bt.pth'

#pretraining the barlow twins way
for epoch in range(n_epoch_barlow):
    len_dataloader = min(len(cataracts_barlow_dataloader), len(d99_barlow_dataloader))

    data_source_barlow_iter = iter(cataracts_barlow_dataloader)
    data_target_barlow_iter = iter(d99_barlow_dataloader)

    for i in range(len_dataloader):

        #barlow twin loss\
        learner.zero_grad()
        source_bt_data,_ = data_source_barlow_iter.next()
        target_bt_data,_ = data_target_barlow_iter.next()
        err_bt_s = learner(source_bt_data[0].to(device), source_bt_data[1].to(device))
        err_bt_t = learner(target_bt_data[0].to(device), target_bt_data[1].to(device))

        err = err_bt_s + err_bt_t
        err.backward()
        optimizer_barlow.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_bt: %f, err_t_bt: %f' \
            % (epoch, i + 1, len_dataloader, err_bt_s.data.cpu().numpy(), err_bt_t.data.cpu().numpy()))
        sys.stdout.flush()

    torch.save(my_net.state_dict(), save_path)


#load the model for the next training with the latest model. (not the best model?)
my_net.load_state_dict(torch.load(save_path))


for p in learner.parameters():
    p.requires_grad = False

for p in my_net.parameters():
    p.requires_grad = True

# training
save_path = './saved_models/src_cata_tgt_d99_model_epoch_current_train_pretrain.pth'
best_accu_t = 0.0
for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source['train']), len(dataloader_target['train']))
    data_source_iter = iter(dataloader_source['train'])
    data_target_iter = iter(dataloader_target['train'])

    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        alpha = 0.5
        # print(alpha)

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size).long()


        s_img = s_img.to(device)
        s_label = s_label.to(device)
        domain_label = domain_label.to(device)


        class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        t_img = t_img.to(device)
        domain_label = domain_label.to(device)

        _, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        
        
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()

        torch.save(my_net.state_dict(), save_path)

    print('\n')
    accu_s = test(save_path, dataloader_source['val'],len(cataracts_class_names))
    print('Accuracy of the %s dataset: %f' % ('cataracts', accu_s))
    accu_t = test(save_path,dataloader_target['val'],len(cataracts_class_names),use_dict=True)
    print('Accuracy of the %s dataset: %f\n' % ('dataset99 balanced', accu_t))
    if accu_t > best_accu_t:
        best_accu_s = accu_s
        best_accu_t = accu_t
        torch.save(my_net.state_dict(), './src_cata_tgt_d99_model_epoch_best_train_pretrain.pth')
