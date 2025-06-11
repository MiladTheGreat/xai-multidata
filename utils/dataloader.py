from torch.utils.data import DataLoader


def get_dataloaders(dataset, ds_cfg,shuffle_train=True):
    
    dataloaders = {}
    params = ds_cfg['params']
    batch_size = params.get('batch_size',32)
    num_workers = params.get('num_workers',4)
    pin_memory = params.get('pin_memory',True)


    for split in dataset: 
        dataloaders[split] = DataLoader(
            dataset=dataset[split],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split=='train' and shuffle_train),
            pin_memory=pin_memory
        )

    return dataloaders

