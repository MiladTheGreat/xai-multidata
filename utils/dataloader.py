from torch.utils.data import DataLoader


def get_dataloaders(dataset, models_cfg,shuffle_train=True):
    
    dataloaders = {}
    
    batch_size = models_cfg.get('batch_size',32)
    num_workers = models_cfg.get('num_workers',4)
    pin_memory = models_cfg.get('pin_memory',True)


    for split in dataset: 
        dataloaders[split] = DataLoader(
            dataset=dataset[split],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split=='train' and shuffle_train),
            pin_memory=pin_memory
        )

    return dataloaders

