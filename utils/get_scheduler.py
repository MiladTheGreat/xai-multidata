import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def get_scheduler(optimizer:Optimizer , num_epoch:int, trainloader_len:int , warmup_percentage:float=0.1,min_lr:float=1e-6):
    
    
    total_steps = num_epoch * trainloader_len
    warmup_steps = int(total_steps*warmup_percentage)
    print(f"scheduler configured for {total_steps} total steps.")
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=min_lr,
        end_factor=1.0,
        total_iters=warmup_steps
        )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max = total_steps - warmup_steps,
        eta_min=min_lr
    )
    
    sequential_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler,main_scheduler],
        milestones=[warmup_steps]
    )
    
    return sequential_scheduler