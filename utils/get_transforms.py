from torchvision import transforms as T
def get_transforms(input_size = 224):
    
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    
    train_transforms = T.Compose([
        T.Resize((int(input_size*1.15),int(input_size*1.15))),
        T.RandomCrop(input_size),
        T.RandomHorizontalFlip(0.5),
        T.RandAugment(num_ops=2,magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.25,scale=(0.02,0.33),ratio=(0.3,3.3),value=0)
    ])

    test_transforms = T.Compose([
        T.Resize(int(input_size*1.15)),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transforms,test_transforms