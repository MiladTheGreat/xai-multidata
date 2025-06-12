import torch.nn as nn
from torchvision import models 
import torchvision
import timm

def get_model_dict(params):


    model_dict = {
        ('swin_t', 'swin_tiny'): [models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1, 'head'],
        ('swin_s', 'swin_small'): [models.swin_s, models.Swin_S_Weights.IMAGENET1K_V1, 'head'],
        ('swin_b', 'swin_base'): [models.swin_b, models.Swin_B_Weights.IMAGENET1K_V1, 'head'],

        ('vit_b_16', 'vit_base_patch16'): [models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, 'heads'],

        ('convnext_t', 'convnext_tiny'): [models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 'classifier'],
        ('convnext_s', 'convnext_small'): [models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 'classifier'],
        ('efficientnet_b7'): [models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1, 'classifier'],
        ('efficientnet_v2_s'): [models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, 'classifier'],

        ('regnet_y_16gf'): [models.regnet_y_16gf, models.RegNet_Y_16GF_Weights.IMAGENET1K_V1, 'fc'],
        ('regnet_y_32gf'): [models.regnet_y_32gf, models.RegNet_Y_32GF_Weights.IMAGENET1K_V1, 'fc'],



        ('deit3_base_patch16_224', 'deit_iii_base'): [
            lambda weights=None: timm.create_model('deit3_base_patch16_224.fb_in1k', pretrained=True if params.get('pretrained',True) else False),
            None,
            'head'
        ],
        
        ('cvt_13', 'cvt13'): [
            lambda weights=None: timm.create_model('cvt_13.in1k', pretrained=True if params.get('pretrained',True) else False),
            None,
            'head'
        ],
        ('cvt_21', 'cvt21'): [
            lambda weights=None: timm.create_model('cvt_21.in1k', pretrained=True if params.get('pretrained',True) else False),
            None,
            'head'
        ],

        ('efficientnetv2_b3', 'tf_efficientnetv2_b3'): [
            lambda weights=None: timm.create_model('tf_efficientnetv2_b3.in21k_ft_in1k', pretrained=True if params.get('pretrained',True) else False),
            None,
            'classifier'
        ],
        
        ('coatnet_0', 'coatnet0'): [
            lambda weights=None: timm.create_model('coatnet_0_224.sw_in1k', pretrained=True if params.get('pretrained',True) else False),
            None,
            'head'
        ],
        ('coatnet_1', 'coatnet1'): [
            lambda weights=None: timm.create_model('coatnet_1_224.sw_in1k', pretrained=True if params.get('pretrained',True) else False),
            None,
            'head'
        ],
        
        ('beitv2_base', 'beit_v2_base'): [
            lambda weights=None: timm.create_model('beitv2_base_patch16_224.in1k_ft_in22k_in1k', pretrained=True if params.get('pretrained',True) else False),
            None,
            'head'
        ],
    }

    return model_dict

def freeze_model(model:nn.Module,head_name:str):

    for param in model.parameters():
        param.requires_grad = False
    classifier = getattr(model,head_name)
    for param in classifier.parameters():
        param.requires_grad = True
    return model    



def get_models(model_name:str,num_classes:int,params:dict):

    model_info = None
    model_dict = get_model_dict(params)
    for names , info in model_dict.items():
        if model_name.lower() in names:
            model_info = info
        
    if model_info is None:
        raise ValueError(f'{model_name} is not in out model list, you can add it in model_dict in model.py')

    model_constructor,weights,head_name = model_info

    net = model_constructor(weights=weights)

    original_classifier = getattr(net,head_name)
    try:
        if isinstance(original_classifier,nn.Linear):
            in_features = original_classifier.in_features

        elif isinstance(original_classifier,nn.Sequential):
            for layer in reversed(original_classifier):
                if isinstance(layer,nn.Linear):
                    in_features = layer.in_features
                    break
            else:
                raise ValueError(f'No nn.Linear found in {model_name}.')
        else:
            in_features = original_classifier.in_features

        print(f'original classifier for {model_name}, in_features:{in_features}')
        if isinstance(original_classifier, nn.Sequential):
            layers = list(original_classifier.children())[:-1]
            layers.append(nn.Linear(in_features, num_classes))
            new_classifier = nn.Sequential(*layers)

        else:
            new_classifier = nn.Linear(in_features,num_classes)
            
        setattr(net,head_name,new_classifier)
        print(f'replaced original classifier. new outupt {num_classes}.')
    except AttributeError:
        raise AttributeError(f'model_does not have attribute named: {head_name}.')
    
    if params.get('freeze',False):
        net = freeze_model(net,head_name)

    return net









