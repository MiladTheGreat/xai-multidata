import torch
import torch.nn as nn
from models.model import get_model_dict

def get_classifier_name(model_name,model_params):
    model_dict = get_model_dict(model_params)
    
    classifier = None
    for name, info in model_dict.items():
        if model_name.lower() in name:
            classifier = info[2]
    if  classifier is not None:

        return classifier
    else:
        raise ValueError("Model Not Found!")

def get_optim(model:nn.Module,
              model_name:str,
              optim_config: dict,
              model_params:dict):
    
    freeze = model_params.get('freeze',False)
    optimizer_name = optim_config.get('name','adamw').lower()
    learning_rate = optim_config.get('lr',1e-4)
    weight_decay = optim_config.get('weight_decay',1e-2)
    use_diff_lr = optim_config.get('diff_lr',True) and not freeze


    param_group = []

    if use_diff_lr:
        classifier_head = get_classifier_name(model_name,model_params)
        head_lr_multiplier = optim_config.get('head_lr_multiplier',10.0)

        head_params = []
        backbone_params = []

        for name ,param in model.named_parameters():
            if not param.requires_grad:
                continue
            if classifier_head in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

            
        param_group.append({'params':backbone_params, 'lr':learning_rate * head_lr_multiplier, 'name':'backbone'})
        param_group.append({'params':head_params, 'lr':learning_rate * head_lr_multiplier , 'name':'head'})

    else:
        if freeze:
            print('backbone is frozen , just head gonna learn.')
        else:
            print('model gonna learn with just one fixed lr.')

        trainable_params = filter(lambda p:p.requires_grad,model.parameters())
        param_group.append({'params': trainable_params})
        

    if optimizer_name == 'adamw':
        print(f" creating AdamW optimizer with base lr={learning_rate}, wd={weight_decay}")
        optimizer = torch.optim.AdamW(param_group, lr=learning_rate, weight_decay=weight_decay)
    else:
            raise ValueError(f'Unsupported optimizer: {optimizer_name}.')
        
    return optimizer