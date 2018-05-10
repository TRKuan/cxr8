import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os

disease_categories = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Effusion': 2,
        'Infiltrate': 3,
        'Mass': 4,
        'Nodule': 5,
        'Pneumonia': 6,
        'Pneumothorax': 7,
        'Consolidation': 8,
        'Edema': 9,
        'Emphysema': 10,
        'Fibrosis': 11,
        'Pleural_Thickening': 12,
        'Hernia': 13,
        }

class CXRDataset(Dataset):

    def __init__(self, root_dir, dataset_type = 'train', transform = None):
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError("No such type, must be 'train', 'val', or 'test'")
        
        self.image_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.index_dir = os.path.join(root_dir, dataset_type+'_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None,nrows=1).ix[0, :].as_matrix()[1:9]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)

        
    def __len__(self):
        return int(len(self.label_index)*0.1)

    def __getitem__(self, idx):
        name = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.label_index.iloc[idx, 1:9].as_matrix().astype('int')
        
        # bbox
        bbox = np.zeros([8, 512, 512])
        bbox_valid = np.zeros(14)
        for i in range(8):
            if label[i] == 0:
               bbox_valid[i] = 1
        
        cols = self.bbox_index.loc[self.bbox_index['Image Index']==name]
        if len(cols)>0:
            for i in range(len(cols)):
                bbox[
                    disease_categories[cols.iloc[i, 1]], #index
                    int(cols.iloc[i, 3]/2): int(cols.iloc[i, 3]/2+cols.iloc[i, 5]/2), #y:y+h
                    int(cols.iloc[i, 2]/2): int(cols.iloc[i, 2]/2+cols.iloc[i, 4]/2) #x:x+w
                ] = 1
                bbox_valid[disease_categories[cols.iloc[i, 1]]] = 1
        
        return image, label, name, bbox, bbox_valid
    
    
class CXRDataset_BBox_only(Dataset):

    def __init__(self, root_dir, transform = None): 
        self.image_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.index_dir = os.path.join(root_dir, 'test'+'_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None,nrows=1).ix[0, :].as_matrix()[1:9]
        self.label_index = pd.read_csv(self.index_dir, header=0)
        self.bbox_index = pd.read_csv(os.path.join(root_dir, 'BBox_List_2017.csv'), header=0)
        
    def __len__(self):
        return len(self.bbox_index)

    def __getitem__(self, idx):
        name = self.bbox_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.label_index.loc[self.label_index['FileName']==name].iloc[0, 1:9].as_matrix().astype('int')
            
        
        # bbox
        bbox = np.zeros([8, 512, 512])
        bbox_valid = np.zeros(8)
        for i in range(8):
            if label[i] == 0:
               bbox_valid[i] = 1
        
        cols = self.bbox_index.loc[self.bbox_index['Image Index']==name]
        if len(cols)>0:
            for i in range(len(cols)):
                bbox[
                    disease_categories[cols.iloc[i, 1]], #index
                    int(cols.iloc[i, 3]/2): int(cols.iloc[i, 3]/2+cols.iloc[i, 5]/2), #y:y+h
                    int(cols.iloc[i, 2]/2): int(cols.iloc[i, 2]/2+cols.iloc[i, 4]/2) #x:x+w
                ] = 1
                bbox_valid[disease_categories[cols.iloc[i, 1]]] = 1
        
        return image, label, name, bbox, bbox_valid
