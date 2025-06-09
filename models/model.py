import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        x = self.layer1(x)
        return x
    
def get_model(model_name,input_size,num_classes):
    model_name = model_name.lower()
    if model_name == 'mlp':
        return SimpleMLP(input_size,num_classes)
    else:
        raise ValueError(f"Model:{model_name} is not defined")
    

    