import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from PIL import Image
import time
import os


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
    trans = transforms.Compose([transforms.ToTensor()])
    image_datasets = {x: CXRDataset(label_path[x], data_dir, transform = trans)for x in ['train', 'val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('Training data: {}\nValidation data: {}'.format(dataset_sizes['train'], dataset_sizes['val']))

    class_names = image_datasets['train'].classes
    return dataloders, dataset_sizes, class_names


def weighted_BCELoss(output, target, weights=None):

    output = output.clamp(min=1e-5, max=1-1e-5)
    if weights is not None:
        assert len(weights) == 2

        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)

    return torch.sum(loss)


def train_model(model, optimizer, num_epochs=10):
    batch_size = 24
    since = time.time()
    dataloders, dataset_sizes, class_names = loadData(batch_size)
    best_model_wts = model.state_dict()
    best_auc = []
    best_auc_ave = 0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            outputList = []
            labelList = []
            logLoss = 0
            # Iterate over data.
            for idx, data in enumerate(dataloders[phase]):
                # get the inputs
                inputs = data['image']
                labels = data['label']

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
                    weights = [BP, BN]
                    if use_gpu:
                        weights = torch.FloatTensor(weights).cuda()
                else: weights = None
                #wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                if phase == 'train':
                    inputs, labels = Variable(inputs, volatile=False), Variable(labels, volatile=False)
                else:
                    inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                out_data = outputs.data
                loss = weighted_BCELoss(outputs, labels, weights=weights)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                labels = labels.data.cpu().numpy()
                out_data = out_data.cpu().numpy()
                for i in range(out_data.shape[0]):
                    outputList.append(out_data[i].tolist())
                    labelList.append(labels[i].tolist())

                logLoss += loss.data[0]
                if idx%100==0 and idx!=0:
                    try: iterAuc =  roc_auc_score(np.array(labelList[-100*batch_size:]),
                                                  np.array(outputList[-100*batch_size:]))
                    except: iterAuc = -1
                    print('{} {:.2f}% Loss: {:.4f} AUC: {:.4f}'.format(phase, 100*idx/len(dataloders[phase]), logLoss/(100*batch_size), iterAuc))
                    logLoss = 0


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_auc_ave = roc_auc_score(np.array(labelList), np.array(outputList))
            epoch_auc = roc_auc_score(np.array(labelList), np.array(outputList), average=None)

            print('{} Loss: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_auc_ave, epoch_auc))
            print()
            for i, c in enumerate(class_names):
                print('{}: {:.4f} '.format(c, epoch_auc[i]))
            print()

            # deep copy the model
            if phase == 'val' and epoch_auc_ave > best_auc_ave:
                best_auc = epoch_auc
                best_auc_ave = epoch_auc_ave
                best_model_wts = model.state_dict()
                saveInfo(model)


        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc_ave))
    print()
    for i, c in enumerate(class_names):
        print('{}: {:.4f} '.format(c, epoch_auc[i]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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
        x = self.globalPool(x)
        x = x.view(x.size(0), -1)
        x = self.prediction(x)#14
        return x


def saveInfo(model):
    #save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet50.pth"))


if __name__ == '__main__':
    model = Model()
    optimizer = optim.Adam([
            {'params':model.transition.parameters()},
            {'params':model.globalPool.parameters()},
            {'params':model.prediction.parameters()}],
            lr=3e-5)

    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()

    #model.load_state_dict(torch.load(os.path.join(save_dir, "resnet50.pth")))

    model = train_model(model, optimizer, num_epochs = 5)
    saveInfo(model)

