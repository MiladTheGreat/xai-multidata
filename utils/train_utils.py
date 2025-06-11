import torch
import os 
from tqdm import tqdm

def train(model,dataloader,optimizer,loss_fn,device):
    total_loss = 0
    model.train()
    for images, labels in tqdm(dataloader,desc="Training"):
        images , labels = images.to(device),labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()

    avg_loss = total_loss/len(dataloader)
    return avg_loss


def evaluate(model,dataloader,device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader,desc="Evaluating"):
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs,dim=1)
            correct+= (preds==labels).sum().item()
            total += len(labels)

    acc = correct/total * 100
    return acc


def save_checkpoint(model,model_name,optimizer,epoch,best_acc,folder="checkpoints"):
    os.makedirs(folder,exist_ok=True)
    path = os.path.join(folder,f"{model_name}.pth")
    torch.save(
        {'epoch':epoch,
         'model_state_dict':model.state_dict(),
         'optimizer_state_dict':optimizer.state_dict(),
         'best_acc':best_acc},
         path
    )
    print(f'Best model saved at {path}')




def train_loop(model,model_name,trainloader,testloader,optimizer,loss_fn,device,num_epochs=20,folder="checkpoints",patience=7):
    patience_counter = 0
    best_acc = 0

    for epoch in range(num_epochs):
        train_loss = train(model,trainloader,optimizer,loss_fn,device)
        test_acc = evaluate(model,testloader,device)
        print(f"epoch:{epoch+1}/{num_epochs}: train loss: {train_loss:.4f} , acuracy: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model,model_name,optimizer,epoch,best_acc)
        else:
            patience_counter+=1
        
        if patience_counter>=patience:
            print('early stopping no improvement.')
    print(f'training coplete. best accuracy:{best_acc}')