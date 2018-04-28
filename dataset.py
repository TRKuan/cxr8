from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os


class CXRDataset(Dataset):

    def __init__(self, root_dir, dataset_type = 'train', transform = None):
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError("No such type, must be 'train', 'val', or 'test'")
        
        self.image_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        index_dir = os.path.join(root_dir, 'label_index.csv')
        self.classes = pd.read_csv(index_dir, header=None,nrows=1).ix[0, :].as_matrix()[1:]
        self.label_list = []
        
        # read in the image names
        image_name_list_dir = os.path.join(root_dir, dataset_type+'_list.txt')
        with open(image_name_list_dir) as f:
            image_name_list = f.read().split('\n')
        
        # find the corresponding label
        label_index = pd.read_csv(index_dir, header=0)
                
        # generate the label list
        pos = 0
        for i in range(len(image_name_list)):
            while image_name_list[i] != label_index.ix[pos, 0]:
                pos += 1
                if pos >= len(label_index):
                    raise ValueError("The image does not have a corresponding label")
            label = (image_name_list[i], label_index.ix[pos, 1:].as_matrix().astype('int'))
            self.label_list.append(label)
            if i%1000 == 0:
                print('\rDataset loading {:.2f}%'.format(100*i/len(image_name_list)), end='\r')
        
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        name = self.label_list[idx][0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = self.label_list[idx][1]
        return image, label, name
    
dataset = CXRDataset('dataset', dataset_type='train')
print(dataset[110])
print(dataset[12])
print(dataset[23])
print(dataset[34])