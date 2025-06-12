import os
import yaml
import torch
from datasets import get_dataset
from utils.get_transforms import get_transforms
from utils.dataloader import get_dataloaders
from models.model import get_models
from utils.train_utils import train_loop
from utils.get_optim import get_optim
from utils.ensemble_vote import ensemble
def main():

    with open('configs/config.yaml','r') as f:
        cfg = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_names= [dataset for dataset in cfg['datasets']]
    model_names = [model for model in cfg['models']]
    ds_cfg = cfg['datasets']
    model_cfg = cfg['models']


    for model_name in model_names:

        for dataset_name in dataset_names:
            

            models_path =os.path.join('outputs',f'{model_name}_{dataset_name}','output')
            os.makedirs(models_path,exist_ok=True)

            model_params = model_cfg[model_name].get('params',{})
            optim_config = model_cfg[model_name].get('optim',{})
            ds_params = ds_cfg[dataset_name].get('params', {})

            train_transform , test_transform = get_transforms(model_params.get('img_size',224))
            
            train_dataset = get_dataset(dataset_name,ds_cfg[dataset_name],split='train',transform=train_transform)
            test_dataset = get_dataset(dataset_name,ds_cfg[dataset_name],split='test',transform=test_transform)
            dataset_dict = {'train':train_dataset,'test':test_dataset}
            print(f"Train dataset size: {len(dataset_dict['train'])}, Test dataset size: {len(dataset_dict['test'])}")
            dataloaders = get_dataloaders(dataset_dict,ds_cfg[dataset_name],shuffle_train=True)
            
            
            

            trained_models = os.listdir(models_path)
            if f'{model_name}_{dataset_name}.pth' not in trained_models or model_cfg[model_name].get('force_retrain', False):
                
                print(f"Training Model: {model_name} on {dataset_name} dataset.")
                
                model = get_models(model_name,ds_params['num_classes'],model_params).to(device)
                optimizer = get_optim(model,model_name,optim_config,model_params)
                label_smoothing = cfg['regularization'].get('label_smoothing',0.1)
                loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                
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

            ensemble(dataset_name,ds_params['num_classes'])


if __name__ == '__main__':
    main()
    