import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
import time
import os

use_gpu = torch.cuda.is_available
data_dir = "./images"
save_dir = "./alexnet.pth"
label_path = {'train':"./Train_Label.csv", 'val':"./Val_Label.csv", 'test':"Test_Label.csv"}

class CXRDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_csv = pd.read_csv(csv_file, header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = pd.read_csv(csv_file, header=None,nrows=1).ix[0, :].as_matrix()

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_csv.ix[idx, 0])
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.labels_csv.ix[idx, 1:].as_matrix().astype('float')
        label = torch.from_numpy(label).type(torch.FloatTensor)
        sample = {'image': image, 'label': label}

        return sample

def loadData(batch_size):
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_datasets = {x: CXRDataset(label_path[x], data_dir, transform = trans)for x in ['train', 'val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('Training data: {}\nValidation data: {}'.format(dataset_sizes['train'], dataset_sizes['val']))
    class_names = image_datasets['train'].classes
    return dataloders, dataset_sizes

def train_model(model, optimizer, scheduler, num_epochs=25):
    batch_size = 40
    since = time.time()
    dataloders, dataset_sizes = loadData(batch_size)
    iterNum = int(80/batch_size)
    best_model_wts = model.state_dict()
    best_auc = 0.0
    lossList = []
    aucList = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_auc = 0.0
            totalAUCCount = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs = data['image']
                labels = data['label']

                #calculate weight for loss
                P = 0
                N = 0
                for label in labels:
                    for v in label:
                        if int(v) == 1: N += 1
                        else: P += 1
                try:
                    BP = (P + N)/P
                    BN = (P + N)/N
                    weight = BP/BN
                except:
                    weight = 1.0
                weight = torch.FloatTensor([weight]).cuda()
                #wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                if phase == 'train': tmpIter = iterNum
                else: tmpIter = 1
                for i in range(tmpIter):                

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    out_data = outputs.data
                    criterion = nn.BCEWithLogitsLoss(weight=weight)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.data[0]
                labels = labels.data.cpu().numpy()
                out_data = out_data.cpu().numpy()
                for i in range(out_data.shape[0]):
                    try:
                        running_auc += roc_auc_score(labels[i], out_data[i])
                        totalAUCCount +=1
                    except: pass

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_auc = running_auc / totalAUCCount
            if phase == 'train': lossList.append(epoch_loss)
            aucList[phase].append(epoch_auc)

            print('{} Loss: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, lossList, aucList

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_ft = models.alexnet(pretrained=True)
        for param in self.model_ft.parameters():
            param.requires_gead = False

        self.transition = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.globalPool = nn.Sequential(
            nn.MaxPool2d(16)
        )
        self.prediction = nn.Sequential(
            nn.Linear(256, 14)
        )
    
    def forward(self, x):
        x = self.model_ft.features(x)#256x31x31
        x = self.transition(x)#256x16x16
        x = self.globalPool(x)#246x1x1
        x = x.view(x.size(0), -1)#256
        x = self.prediction(x)#14
        return x



if __name__ == '__main__':
    model = Model()
    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7, gamma=0.1)

    model, lossList, aucList = train_model(model, optimizer, exp_lr_scheduler, num_epochs = 10)
    torch.save(model.state_dict(), save_dir)
    plt.plot(lossList, label='loss', color='blue')
    plt.show()
    plt.plot(aucList['train'], label='auc_train', color='red')
    plt.plot(aucList['val'], label='auc_val', color='green')
    plt.show()

