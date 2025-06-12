from timm.data.mixup import Mixup

def create_mixup_fn(config:dict , num_classes:int):

    aug_config = config.get('augmentation', {})
    mixup_config = aug_config.get('mixup_cutmix', {})

    if not mixup_config.get('enable',False):
        print('Mixup is disabled.')
        return None
    print('Mixup is enabled.')

    mixup_fn = Mixup(
        mixup_alpha=mixup_config.get('mixup_alpha', 0.8),
        cutmix_alpha=mixup_config.get('cutmix_alpha', 1.0),
        prob=mixup_config.get('prob', 1.0),
        switch_prob=mixup_config.get('switch_prob', 0.5),
        mode='batch',
        label_smoothing=config.get('regularization', {}).get('label_smoothing', 0.0),
        num_classes=num_classes
    )
    return mixup_fn