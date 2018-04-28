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
        self.index_dir = os.path.join(root_dir, dataset_type+'_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None,nrows=1).ix[0, :].as_matrix()[1:]
        self.label_index = pd.read_csv(self.index_dir, header=0)

        
    def __len__(self):
        return len(self.label_index)

    def __getitem__(self, idx):
        name = self.label_index.ix[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_index.ix[idx, 1:].as_matrix().astype('int')
        return image, label, name
