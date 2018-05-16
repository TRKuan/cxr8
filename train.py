from dataset import CXRDataset, CXRDataset_BBox_only
from model import Model
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
from tensorboardX import SummaryWriter
import time
import os

batch_size = 4
num_epochs = 100
learning_rate = 1e-6
regulization = 0
model_save_dir = './savedModels'
model_name = 'net_v1_lr_1e-6_bbox_data_arg'
log_dir = './runs'
data_root_dir = './dataset'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.50576189]
def make_dataLoader():
    trans = {}
    trans['train'] = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1.])
    ])
    trans['val'] = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1.])
    ])
    datasets = {
        'train': CXRDataset_BBox_only(data_root_dir, transform=trans['train'], data_arg=True),
        'val': CXRDataset(data_root_dir, dataset_type='val', transform=trans['val'])
    }
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = datasets['train'].classes

    print(dataset_sizes)
    
    return dataloaders, dataset_sizes, class_names

def weighted_BCELoss(output, target, weights=None):
    output = output.clamp(min=1e-5, max=1-1e-5)
    target = target.float()
    if weights is not None:
        assert len(weights) == 2

        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)

    return torch.sum(loss)

def training(model):
    writer = {x: SummaryWriter(log_dir=os.path.join(log_dir, model_name, x),
                comment=model_name)
          for x in ['train', 'val']}
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)
    dataloaders, dataset_sizes, class_names = make_dataLoader()
    
    since = time.time()
    best_model_wts = model.state_dict()
    best_auc = []
    best_auc_ave = 0.0
    iter_num = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #scheduler.step()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            output_list = []
            label_list = []
            
            # Iterate over data.
            for idx, data in enumerate(dataloaders[phase]):
                # get the inputs
                images, labels, names, bboxes, bbox_valids = data

                images = images.to(device)
                labels = labels.to(device)
                
                if phase == 'train':
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)
                    
                #calculate weight for loss
                P = 0
                N = 0
                for label in labels:
                    for v in label:
                        if int(v) == 1: P += 1
                        else: N += 1
                if P!=0 and N!=0:
                    BP = (P + N)/P
                    BN = (P + N)/N
                    weights = torch.tensor([BP, BN], dtype=torch.float).to(device)
                else: weights = None

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, segs = model(images)
                
                # remove invalid bbox and segmentation outputs
                bbox_list = []
                for i in range(bbox_valids.size(0)):
                    bbox_list.append([])
                    for j in range(8):
                        if bbox_valids[i][j] == 1:
                            bbox_list[i].append(bboxes[i][j])
                    bbox_list[i] = torch.stack(bbox_list[i]).to(device)
                
                seg_list = []
                for i in range(bbox_valids.size(0)):
                    seg_list.append([])
                    for j in range(8):
                        if bbox_valids[i][j] == 1:
                            seg_list[i].append(segs[i][j])
                    seg_list[i] = torch.stack(seg_list[i]).to(device)
                
                # classification loss
                loss = weighted_BCELoss(outputs, labels, weights=weights)
                # segmentation loss
                for i in range(len(seg_list)):
                    loss += 5*weighted_BCELoss(seg_list[i], bbox_list[i], weights=torch.tensor([10., 1.]).to(device))/(512*512)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    iter_num += 1

                # metrix
                running_loss += loss.item()
                outputs = outputs.detach().to('cpu').numpy()
                labels = labels.detach().to('cpu').numpy()
                for i in range(outputs.shape[0]):
                    output_list.append(outputs[i].tolist())
                    label_list.append(labels[i].tolist())
                    
                if idx%10 == 0:
                    if phase == 'train':
                        writer[phase].add_scalar('loss', loss.item()/outputs.shape[0], iter_num)
                    print('\r{} {:.2f}%'.format(phase, 100*idx/len(dataloaders[phase])), end='\r')
                if idx%100 == 0 and idx!=0:
                    if phase == 'train':
                        try:
                            auc = roc_auc_score(np.array(label_list[-100*batch_size:]), np.array(output_list[-100*batch_size:]))
                            writer[phase].add_scalar('auc', auc, iter_num)
                        except:
                            pass

            epoch_loss = running_loss / dataset_sizes[phase]
            try:
                epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))
                epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list), average=None)
            except:
                epoch_auc_ave = 0
                epoch_auc = [0 for _ in range(len(class_names))]

            if phase == 'val':
                writer[phase].add_scalar('loss', epoch_loss, iter_num)
                writer[phase].add_scalar('auc', epoch_auc_ave, iter_num)
            for i, c in enumerate(class_names):
                writer[phase].add_pr_curve(c, np.array(label_list[:][i]), np.array(output_list[:][i]), iter_num)
            
            log_str = ''
            log_str += '{} Loss: {:.4f} AUC: {:.4f}  \n\n'.format(
                phase, epoch_loss, epoch_auc_ave, epoch_auc)
            for i, c in enumerate(class_names):
                log_str += '{}: {:.4f}  \n'.format(c, epoch_auc[i])
            log_str += '\n'
            print(log_str)
            writer[phase].add_text('log',log_str , iter_num)

            # save model
            if phase == 'val' and epoch_auc_ave > best_auc_ave:
                best_auc = epoch_auc
                best_auc_ave = epoch_auc_ave
                best_model_wts = model.state_dict()
                model_dir = os.path.join(model_save_dir, model_name+'.pth')
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(model.state_dict(), model_dir)
                print('Model saved to %s'%(model_dir))
                writer[phase].add_text('log','Model saved to %s\n\n'%(model_dir) , iter_num)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc_ave))
    print()
    for i, c in enumerate(class_names):
        print('{}: {:.4f} '.format(c, best_auc[i]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

if __name__ == '__main__':
    model = Model().to(device)
    training(model)
