def get_dataset(name,config,split='train',transform=None,**kwargs):
    name = name.lower()
    root = config['root']
    
    if name =='cub':
        from .cub200 import CUB200Dataset
        return CUB200Dataset(root,split,transform)
    elif name == 'folder':
        from .folder import FolderDataset
        return FolderDataset(root, split,transform)
    else:
        raise ValueError(f'Dataset {name} is not supported!')