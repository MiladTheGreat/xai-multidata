import torch.nn as nn
from torchvision import models 

    
def get_model(model_name,num_classes,params=None):
    model_name = model_name.lower()
    params = params or {}

    if model_name in ['swin_t','swin_tiny']:
        net = models.swin_t(weights="IMAGENET1K_V1" if params.get('pretrained',True) else None)
        net.head = nn.Linear(net.head.in_features,num_classes)
        if params.get('freeze',False):
            for name,param in net.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        return net
    

    elif model_name.startswith('efficientnet_v2_'):
        net = getattr(models,model_name)(weights= "IMAGENET1K_V1" if params.get('pretrained',True) else None)
        net.classifier[1] = nn.Linear(net.net.classifier[1].in_features,num_classes)
        if params.get('freeze',False):
            for name,param in net.named_parameters():
                if 'classifier.1' not in name:
                    param.requires_grad = False

        return net
    
    elif model_name.startswith('efficientnet_'):
        net = getattr(models, model_name)(weights="IMAGENET1K_V1" if params.get('pretrained', True) else None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        if params.get('freeze', False):
            for name, param in net.named_parameters():
                if "classifier.1" not in name:
                    param.requires_grad = False
        return net
    
    elif model_name in ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']:
        net = getattr(models, model_name)(weights="IMAGENET1K_V1" if params.get('pretrained', True) else None)
        net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
        if params.get('freeze', False):
            for name, param in net.named_parameters():
                if "heads.head" not in name:
                    param.requires_grad = False
        return net
    
    else:
        raise ValueError(f"Model '{model_name}' is not implemented.")   