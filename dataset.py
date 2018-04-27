from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os


class CXRDataset(Dataset):

    def __init__(self, index_dir, image_dir, type = 'train', transform = None):
        self.index_dir = index_dir
        self.image_dir = image_dir
        self.label_index = pd.read_csv(index_dir, header=0)
        self.transform = transform
        self.classes = pd.read_csv(index_dir, header=None,nrows=1).ix[0, :].as_matrix()[1:]

    def __len__(self):
        return len(self.label_index)

    def __getitem__(self, idx):
        img_dir = os.path.join(self.image_dir, self.label_index.ix[idx, 0])
        image = Image.open(img_dir).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_index.ix[idx, 1:].as_matrix().astype('int')
        return image, label

if __name__ == '__main__':
    index_dir = 'dataset/label_index.csv'
    image_dir = 'dataset/images'
    dataset = CXRDataset(index_dir, image_dir)
    print(dataset[0])
