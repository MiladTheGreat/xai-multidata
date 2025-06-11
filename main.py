import os
import yaml
import torch
from datasets import get_dataset
from utils.get_transforms import get_transforms
from utils.dataloader import get_dataloaders
from models.model import get_model
from utils.train_utils import train_loop


with open('configs/config.yaml','r') as f:
    cfg = yaml.safe_load(f)

datasets_cfg = cfg['dataset']
models_cfg = cfg['model']
train_cfg = cfg['train']
optim_cfg = cfg['optim']
checkpoint_cfg = cfg['checkpoint']
output_cfg = cfg['output']


train_transform , test_transform = get_transforms()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = get_dataset(datasets_cfg['name'],root=datasets_cfg['root'],split='train',transform=train_transform)
test_dataset = get_dataset(datasets_cfg['name'],root=datasets_cfg['root'],split='test',transform=test_transform)
dataset_dict = {'train':train_dataset,'test':test_dataset}
print(f"Train dataset size: {len(dataset_dict['train'])}, Test dataset size: {len(dataset_dict['test'])}")

dataloaders = get_dataloaders(dataset_dict,models_cfg,shuffle_train=True)



params = models_cfg.get('params',{})
model = get_model(models_cfg).to(device)
print(type(model))
optimizer = torch.optim.Adam(model.parameters(),optim_cfg['lr'])
loss_fn = torch.nn.CrossEntropyLoss()


trained_models = os.listdir('checkpoints')


if f'{models_cfg['name']}.pth' not in trained_models:

    train_loop(model,
               models_cfg['name'],
               dataloaders['train'],
               dataloaders['test'],
               optimizer,
               loss_fn,device,
               train_cfg['epoch'],
               folder=checkpoint_cfg['dir'],
               patience=train_cfg['patience'])


