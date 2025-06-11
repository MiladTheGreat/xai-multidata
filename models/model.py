import torch.nn as nn
from torchvision import models 
import torchvision
    

def get_model(cfg):
    model_name = cfg['name']
    num_classes = cfg['num_classes']
    params = cfg['params'] or {}
    model_name = model_name.lower()

    if model_name in ['swin_t','swin_tiny']:
        net = models.swin_t(weights=( models.Swin_T_Weights.IMAGENET1K_V1 if params.get('pretrained',True) else None))
        net.head = nn.Linear(net.head.in_features,num_classes)
        if params.get('freeze',False):
            for name , param in net.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        return net
    
    elif model_name.statswith('efficientnet_v2_'):
        net = getattr(models,model_name)(weights = ('IMAGENET1K_V1' if params.get('pretrained',True) else None))
        net.classifier[1] = nn.Linear(net.classifier[1].in_features,num_classes)
        if params.get('freeze',False):
            for name , param in net.named_parameters():
                if 'classifier.1' not in name:
                    param.requires_grad = False

    elif model_name.statswith('efficientnet_'):
        net = getattr(models,model_name)(weights = ('IMAGENET1K_V1' if params.get('pretrained',True) else None))
        net.classifier[1] = nn.Linear(net.classifier[1].in_features,num_classes)
        if params.get('freeze',False):
            for name , param in net.named_parameters():
                if 'classifier.1' not in name:
                    param.requires_grad = False

    elif model_name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
        net = getattr(models,model_name)(weights= ('IMAGENET1K_V1' if params.get('pretrained',True) else None))
        net.heads.head = nn.Linear(net.heads.head.in_features,num_classes)
        if params.get('freeze',False):
            for name,param in net.named_parameters():
                if 'head' not in name:
                    param.requires_grad =False

    else:
        raise ValueError(f'No model named {model_name} found!')














