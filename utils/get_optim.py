import torch.optim as optim

def get_optim(optim_name):
    optim_name = optim_name.lower()
    optim_dict = {
        'sgd':      optim.SGD,     
        'sgd_m':    optim.SGD,     
        'rmsprop':  optim.RMSprop,
        'adagrad':  optim.Adagrad,
        'adadelta': optim.Adadelta,
        'adam':     optim.Adam,
        'adamw':    optim.AdamW,   
        'adamax':   optim.Adamax,
        'lbfgs':    optim.LBFGS,
        'nadam':    optim.NAdam,   
        'adamams':  optim.ASGD,    
    }
    return optim_dict.get(optim_name,optim.Adam)
