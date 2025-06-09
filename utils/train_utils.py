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


def save_checkpoint(model,model_name,folder="checkpoints"):
    os.makedirs(folder,exist_ok=True)
    path = os.path.join(folder,f"{model_name}.pth")
    torch.save(model.state_dict(),path)
    print(f'Best model saved at {path}')