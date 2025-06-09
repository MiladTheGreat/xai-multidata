from datasets.base import BaseDataset
import os 
import pandas as pd
from PIL import Image

class CUB200Dataset(BaseDataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)
        self._load_dataset()

    def _load_dataset(self):
        img_paths  = pd.read_csv(os.path.join(self.root,'images.txt'),sep=' ',names=['img_id','path'])
        img_labels = pd.read_csv(os.path.join(self.root,'image_class_labels.txt'),sep = ' ',names = ['img_id','label'])
        train_test_split = pd.read_csv(os.path.join(self.root,'train_test_split.txt'),sep=' ',names=['img_id','train'])
        df = pd.merge(img_paths,img_labels,on='img_id')
        df = pd.merge(df,train_test_split,on='img_id')
        if self.split.lower() == 'train':
            self.df = df[df['train']==1].drop('train',axis=1)
        elif self.split.lower() == 'train':
            self.df = df[df['train']==0].drop('train',axis=1)
        else:
            self.df= df 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        i = self.df.iloc[idx]
        label = i['label'] - 1
        path = os.path.join(self.root,'images',i['path'])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img,label
