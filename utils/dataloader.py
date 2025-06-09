from torch.utils.data import DataLoader


def get_dataloaders(dataset, batch_size=32 , num_workers=4, shuffle_train=True):
    
    dataloaders = {}
    
    for split in dataset: 

        dataloaders[split] = DataLoader(
            dataset=dataset[split],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split=='train' and shuffle_train),
            pin_memory=True
        )

    return dataloaders

