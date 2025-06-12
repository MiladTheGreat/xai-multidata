import torch
import os 
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from utils.get_scheduler import get_scheduler
from utils.get_mixup import create_mixup_fn
def train(model,dataloader,optimizer,loss_fn,device,mixup_fn):
    model.train()
    total_loss = 0

    for images, labels, paths in tqdm(dataloader,desc="Training"):
        images , labels = images.to(device),labels.to(device)
        if mixup_fn:
            images, labels = mixup_fn(images, labels)
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()

    avg_loss = total_loss/len(dataloader)
    return avg_loss


def evaluate(model,dataloader,loss_fn,device):
    model.eval()
    correct = 0
    total_count = 0
    total_loss = 0 
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader,desc="Evaluating"):
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs,labels)  
            total_loss+= loss.item()
            preds = torch.argmax(outputs,dim=1)
            correct+= (preds==labels).sum().item()
            total_count += len(labels)

    acc = correct/total_count * 100
    avg_loss = total_loss/len(dataloader)
    return avg_loss,acc


def save_checkpoint(model,model_name,dataset_name,optimizer,epoch,best_acc,path):
    folder_path = os.path.join(path,'output')
    os.makedirs(folder_path,exist_ok=True)
    path = os.path.join(folder_path,f"{model_name}.pth")
    torch.save(
        {'epoch':epoch,
         'model_state_dict':model.state_dict(),
         'optimizer_state_dict':optimizer.state_dict(),
         'best_acc':best_acc},
         path
    )
    print(f'Best model saved at {path}')

def save_predictions(model,dataloader,device,save_path):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader,desc=f"logging the {os.path.basename(save_path)}"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs,dim=1).cpu().numpy()
            confidence = F.softmax(outputs,dim=1).max(dim=1)[0].cpu().numpy()
            percentage = F.softmax(outputs,dim=1).cpu().numpy()  
            labels = labels.cpu().numpy()
            for i,(img_path,gt,pred,conf) in enumerate(zip(paths,labels,preds,confidence)):
                specific_percentage = percentage[i]
                rows.append({
                    'image_path':os.path.basename(img_path),
                    'label':int(gt),
                    'prediction':int(pred),
                    'confidence':float(conf),
                    'percentage':specific_percentage.tolist()
                })

    try:    
        pd.DataFrame(rows).to_parquet(save_path,index=False,engine='pyarrow')
    except:
        save_path = save_path[:-4]+'_1.parquet'
        pd.DataFrame(rows).to_parquet(save_path,index=False,engine='pyarrow')
        print(f'csv file was open, new_name is {os.path.basename(save_path)}')


def epoch_logger(epoch,train_loss,val_loss,val_acc,log_path,):
    csv_path = os.path.join(log_path,f"epoch_logs.csv")
    new_row = [{'epoch':epoch+1,
                'train_loss':train_loss,
                'val_loss':val_loss,
                'val_acc':val_acc}]
    if os.path.isfile(csv_path):
        pd.DataFrame(new_row).to_csv(csv_path,mode='a',header=False,index=False)
    else:
        pd.DataFrame(new_row).to_csv(csv_path,mode='a',header=True,index=False)



def train_loop(model,model_name,trainloader,testloader,optimizer,loss_fn,device,dataset_name,cfg):
    patience_counter = 0
    best_acc = 0
    num_epochs = cfg['train']['epochs']
    patience = cfg['train']['patience']
    log_path = cfg['output']['dir']
    ds_params = cfg['datasets'][dataset_name]['params']
    model_name = f'{model_name}_{dataset_name}'
    log_path = os.path.join(log_path,model_name)
    os.makedirs(log_path,exist_ok=True)
    scheduler = get_scheduler(optimizer,num_epochs,len(trainloader))
    mixup_fn = create_mixup_fn(cfg, num_classes=ds_params['num_classes'])
    for epoch in range(num_epochs):
        train_loss= train(model,trainloader,optimizer,loss_fn,device,mixup_fn)
        scheduler.step()
        test_loss,test_acc = evaluate(model,testloader,loss_fn,device)
        epoch_logger(epoch,train_loss,test_loss,test_acc,log_path)
        print(f"epoch:{epoch+1}/{num_epochs}: train loss: {train_loss:.4f} , acuracy: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model,model_name,dataset_name,optimizer,epoch,best_acc,log_path)
        else:
            patience_counter+=1
        
        if patience_counter>=patience:
            print('early stopping no improvement.')
        


    save_predictions(model,testloader,device,os.path.join(log_path,f"preds_val_{dataset_name}_dataset.parquet"))
    print(f'training coplete. best accuracy:{best_acc:.2f}')
