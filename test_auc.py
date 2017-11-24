import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
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
import logging


use_gpu = torch.cuda.is_available
data_dir = "./images"
save_dir = "./savedModels"
log_dir = "./log"
statistic_dir = "./statistic"
label_path = {'train':"./Train_Label.csv", 'val':"./Val_Label.csv", 'test':"Test_Label.csv"}

numtolabel = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
       'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
       'Pleural Thickening', 'Hernia'
      ]

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
    image_datasets = {x: CXRDataset(label_path[x], data_dir, transform = trans)for x in ['val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                  for x in ['val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

    return dataloders, dataset_sizes

def test_model(model):
    batch_size = 4 
    since = time.time()
    dataloders, dataset_sizes = loadData(batch_size)
    iterNum = int(80/batch_size)
    best_model_wts = model.state_dict()
    best_auc = 0.0
    lossList = []
    aucList = {'train': [], 'val': []}
    lastAUC = 0
    outputList = []
    labelList = []

    for epoch in range(1):


        # Each epoch has a training and validation phase
        for phase in ['val']:


            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs = data['image']
                labels = data['label']

                #wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                

                # forward
                outputs = model(inputs)
                out_data = outputs.data
                # statistics
                labels = labels.data.cpu().numpy()
                out_data = out_data.cpu().numpy()
                for i in range(out_data.shape[0]):
                    outputList.append(out_data[i].tolist())
                    labelList.append(labels[i].tolist())

    fig = plt.figure()
    plt.suptitle("roc curve")

    for i in range(14):
        o_list = []
        l_list = []
        for n in range(len(outputList)):
            o_list.append(outputList[n][i])
            l_list.append(labelList[n][i])
        try:
            auc = roc_auc_score(l_list, o_list)
            roc = roc_curve(l_list, o_list)
            print('{} auc:{:.4f}'.format(numtolabel[i], auc))
            plt.plot(roc.tpr, roc.fpr, label=numtolabel[i])
            plt.legend()


        except: print('{} error'.format(i))
    plt.legend()
    plt.ylabel("tpr")
    plt.xlabel("fpr")
    plt.legend()
    plt.show()

    return model, lossList, aucList

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_ft = models.alexnet(pretrained=True)
        self.model_ft = nn.Sequential(*list(self.model_ft.features.children())[:-1])
        for param in self.model_ft.parameters():
            param.requires_gead = False

        self.transition = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
            #nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.globalPool = nn.Sequential(
            nn.MaxPool2d(63)
        )
        self.prediction = nn.Sequential(
            nn.Linear(256, 14)
        )
 
    def forward(self, x):
        x = self.model_ft(x)#256x31x31
        out_trans = self.transition(x)#256x16x16
        out = self.globalPool(out_trans)#246x1x1
        out = out.view(out.size(0), -1)#256
        out = self.prediction(out)#14
        return out

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
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
    model = Model_1()
    model.load_state_dict(torch.load(os.path.join(save_dir, "alexnet.pth")))
    if use_gpu:
        model = model.cuda()
    test_model(model)

