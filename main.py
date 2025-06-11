import os
import yaml
import torch
from datasets import get_dataset
from utils.get_transforms import get_transforms
from utils.dataloader import get_dataloaders
from models.model import get_model
from utils.train_utils import train_loop

def main():
    with open('configs/config.yaml','r') as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform , test_transform = get_transforms()
    dataset_names= [dataset for dataset in cfg['datasets']]
    model_names = [model for model in cfg['models']]
    ds_cfg = cfg['datasets']
    model_cfg = cfg['models']
    for dataset_name in dataset_names:

        train_dataset = get_dataset(dataset_name,ds_cfg[dataset_name],split='train',transform=train_transform)
        test_dataset = get_dataset(dataset_name,ds_cfg[dataset_name],split='test',transform=test_transform)
        dataset_dict = {'train':train_dataset,'test':test_dataset}
        print(f"Train dataset size: {len(dataset_dict['train'])}, Test dataset size: {len(dataset_dict['test'])}")

        dataloaders = get_dataloaders(dataset_dict,ds_cfg[dataset_name],shuffle_train=True)

    


        for model_name in model_names:
            print(f"Training Model: {model_name}")
            params = model_cfg[model_name].get('params',{})
            model = get_model(model_name,model_cfg[model_name],num_classes= ds_cfg[dataset_name]['params']['num_classes']).to(device)
            optimizer = torch.optim.Adam(model.parameters(),cfg['optim']['lr'])
            loss_fn = torch.nn.CrossEntropyLoss()


            trained_models = os.listdir('checkpoints')


            if f'{model_name}_{dataset_name}.pth' not in trained_models or model_cfg[model_name].get('force_retrain', False):

                train_loop(model,
                        model_name,
                        dataloaders['train'],
                        dataloaders['test'],
                        optimizer,
                        loss_fn,
                        device,
                        dataset_name,
                        cfg)
            else:
                print(f'{model_name} is already trained on {dataset_name} dataset, for re-training change the value of force_retrain in config to True.')


if __name__ == '__main__':
    main()