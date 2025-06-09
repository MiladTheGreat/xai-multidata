from torchvision.datasets import ImageFolder
from datasets.base import BaseDataset
from torch.utils.data import random_split

class FolderDataset(BaseDataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)
        
        dataset = ImageFolder(self.root,transform=self.transform)

        data_set_size = len(dataset)
        train_size = int(0.8 * (data_set_size))
        test_size  = data_set_size - train_size
        self.train , self.test = random_split(dataset,[train_size,test_size])
        if split.lower() == 'train':
            self.final_dataset = self.train
        elif split.lower() == 'test':
            self.final_dataset = self.test
        else:
            raise ValueError(f'Unknown split: {split}')

    def __len__(self):
        return len(self.final_dataset)
    
    def __getitem__(self, idx):
        return self.final_dataset[idx]