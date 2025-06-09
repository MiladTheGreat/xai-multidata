from torchvision import transforms as T
def get_transforms():
    train_transforms = T.Compose([
        T.RandomResizedCrop(224,scale=(0.8,1)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness = 0.3,contrast=0.8,saturation=0.3,hue=0.3),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(0.5,scale=(0.02,0.1),ratio=(0.3,3.3),value=0,inplace=False)
    ])

    test_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms,test_transforms