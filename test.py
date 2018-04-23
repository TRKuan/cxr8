import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
import time
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2

use_gpu = torch.cuda.is_available
data_dir = "./images"
save_dir = "./savedModels"
label_path = {'train':"./Train_Label.csv", 'val':"./Val_Label.csv", 'test':"Test_Label.csv"}

class CXRDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform = None):
        self.labels_csv = pd.read_csv(csv_file, header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = pd.read_csv(csv_file, header=None,nrows=1).ix[0, :].as_matrix()
        self.classes = self.classes[1:]

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_csv.ix[idx, 0])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = self.labels_csv.ix[idx, 1:].as_matrix().astype('float')
        label = torch.from_numpy(label).type(torch.FloatTensor)
        sample = {'image': image, 'label': label}

        return sample

def loadData(batch_size):
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_datasets = {x: CXRDataset(label_path[x], data_dir, transform = trans)for x in ['test']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4)
                  for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    print('Testing data: {}'.format(dataset_sizes['test']))

    class_names = image_datasets['test'].classes
    return dataloders, dataset_sizes, class_names

def test_model(model):
    batch_size = 24
    since = time.time()
    dataloders, dataset_sizes, class_names = loadData(batch_size)


    model.train(False)  # Set model to evaluate mode

    outputList = []
    labelList = []
    num=0
    # Iterate over data.
    for idx, data in enumerate(dataloders['test']):
        # get the inputs
        inputs = data['image']
        labels = data['label']

        #wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
                
        outputs = model(inputs)
        out_data = outputs.data

        labels = labels.data.cpu().numpy()
        out_data = out_data.cpu().numpy()
        for i in range(out_data.shape[0]):
            outputList.append(out_data[i].tolist())
            labelList.append(labels[i].tolist())
        
        if idx%20 == 0 and idx!=0:
            print('\r{:.2f}%'.format(100*idx/len(dataloders['test'])), end='')    
    print()
    '''
    plt.figure()
    heatmap = heatmap.data.cpu().numpy()[:, :, 1]
    heatmap = cv2.resize(heatmap, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
    plt.imshow(heatmap)
    plt.show()
    '''
    print('Sample outputs-----------------')
    rand_list = []
    while len(rand_list) < 30:
        rand = random.randint(0, dataset_sizes['test'])
        if rand not in rand_list:
            rand_list.append(rand)
    for i in rand_list:
        print('{} output: {}\nlabel: {}\n---------------'.format(i, ['{:.4f}'.format(item) for item in outputList[i]], labelList[i]))
    
    epoch_auc_ave = roc_auc_score(np.array(labelList), np.array(outputList))
    epoch_auc = roc_auc_score(np.array(labelList), np.array(outputList), average=None)
    

    print('AUC: {:.4f}'.format(epoch_auc_ave))
    print()
    plt.figure()
    for i, c in enumerate(class_names):
        fpr, tpr, _ = roc_curve(np.array(labelList)[:, i], np.array(outputList)[:, i])
        plt.plot(fpr, tpr, label=c)
        print('{}: {:.4f} '.format(c, epoch_auc[i]))
    print()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        for param in self.model_ft.parameters():
            param.requires_grad = False

        self.transition = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.globalPool = nn.Sequential(
            nn.MaxPool2d(32)
        )
        self.prediction = nn.Sequential(
            nn.Linear(2048, 14),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)


        x = self.transition(x)
        active = x
        x = self.globalPool(x)
        x = x.view(x.size(0), -1)
        x = self.prediction(x)#14
        #for name, p in self.prediction.named_parameters():
        #    if name == '0.weight': weight = p
        return x#, torch.matmul(active.view(2048, 32, 32).permute(1, 2, 0), weight.permute(1, 0))


if __name__ == '__main__':
    model = Model()
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, "resnet50.pth"))) 
    model = test_model(model.cuda())

