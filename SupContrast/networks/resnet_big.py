import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
   
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x
    
class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, pretrained=False):
        super(SupConResNet, self).__init__()
        self.encoder =  torchvision.models.resnet50(pretrained=False)
        
        dim_in = self.encoder.fc.in_features 
        
        if pretrained:
            #set the simclr encoder pre-trained on imagenet
            cpt = torch.load('/experimentos/pesos/simclr/simclr_r50_800.pth')
            msg = self.encoder.load_state_dict(cpt)
            print(15*"**")
            print(f"MODEL LOADED WITH MSG: {msg}")
            print(15*"**")
        
        self.encoder.fc = Identity()
        

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


