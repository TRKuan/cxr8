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
import logging


use_gpu = torch.cuda.is_available
data_dir = "./images"
save_dir = "./savedModels"
log_dir = "./log"
statistic_dir = "./statistic"
label_path = {'train':"./Train_Label.csv", 'val':"./Val_Label.csv", 'test':"Test_Label.csv"}

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename=os.path.join(log_dir, "alxnetlog.log"), filemode='w', level=logging.INFO, format='%(message)s')

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
    batch_size = 1 
    since = time.time()
    dataloders, dataset_sizes = loadData(batch_size)
    iterNum = int(80/batch_size)
    best_model_wts = model.state_dict()
    best_auc = 0.0
    lossList = []
    aucList = {'train': [], 'val': []}
    lastAUC = 0

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
                outputs, trans_out = model(inputs)
                labels = labels.data.cpu().numpy()[0]
                trans_out = trans_out.data.cpu().numpy()[0].transpose(1, 2, 0)
                for layer in model.modules():
                    if isinstance(layer, nn.Linear):
                         pred_weight=layer.weight.data.cpu().numpy().transpose(1, 0)

                heatmap = np.matmul(trans_out, pred_weight)
                for i in range(14):
                    if labels[i] == 1:
                        image = heatmap.transpose(2, 0, 1)[i]
                        image *= 255
                        for row in image:
                            for px in row:
                                if px < 60: px = 0
                                if px > 180: px = 0
                        plt.imshow(image)
                        plt.show()

                '''
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
                '''
                 
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
        out_trans = self.transition(x)#256x16x16
        out = self.globalPool(out_trans)#246x1x1
        out = out.view(out.size(0), -1)#256
        out = self.prediction(out)#14
        return x, out_trans



if __name__ == '__main__':
    model = Model()
    model.load_state_dict(torch.load(os.path.join(save_dir, "alexnet.pth")))
    if use_gpu:
        model = model.cuda()
    test_model(model)

