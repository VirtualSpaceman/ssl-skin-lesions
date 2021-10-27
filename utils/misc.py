import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import models
from utils import build_backbone_infomin as bbi
import numpy as np
import os



class Identity(nn.Module):
    """
    Identity layer 

    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class AugmentOnTest:
    """
        wrapper to assess test samples

    """
    
    def __init__(self, dataset, n):
        """
        Args:
        dataset: a instance of torch.utils.data.Dataset class.

        n: number of copies to be created for each sample.

        """
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n * len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i // self.n]

def get_model(method, ft_from_featuremap):
    """
    function to get the pre-trained network given the method as input

    Args:
    method: which method to return the pre-trained network

    ft_from_featuremap: indicates if fine-tuning from feature map
                        or latent representation

    """
    print(30*"**")
    print(f"Loading pre-trained weights from {method}...")

    resnet = None
    msg = "Empty message"

    if method == 'simclr':
        print("Loading SimCLR...")
        #weights from https://github.com/google-research/simclr
        cpt = torch.load('/experimentos/pesos/simclr/simclr_r50_800.pth')
        resnet = models.resnet50(pretrained=False)
        msg = resnet.load_state_dict(cpt)
        resnet.fc = Identity()
    elif method == 'swav':
        #weights from https://github.com/facebookresearch/swav
        resnet = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        resnet.fc = Identity()
    elif method == 'baseline':
        resnet = models.resnet50(pretrained=True)
        resnet.fc = Identity()
    elif method == 'byol':
        #weights from https://github.com/deepmind/deepmind-research/tree/master/byol
        cpt = torch.load('/experimentos/pesos/byol/byol_res50x1.pth.tar')
        resnet = models.resnet50(pretrained=False)
        msg = resnet.load_state_dict(cpt)
        resnet.fc = Identity()
    elif method == 'infomin':
        model, _ = bbi.build_model()
        model, msg = bbi.load_encoder_weights(model)
        resnet = model.encoder
    elif method == 'moco':
        #weights from https://github.com/facebookresearch/moco
        path = '/experimentos/pesos/mocov2/moco_v2_800ep_pretrain.pth.tar'
        resnet, msg = load_moco(path)
        resnet.fc = Identity()
    else:
        raise NotImplementedError()
    
    
    print(30*"**")
    print("Loading Weights with message: ",msg)
    print(30*"**")
    assert resnet is not None, "Encoder Network is None. Fix this issue."
    #adjust the last layer if necessary
    resnet = adjust_last_layers(method, ft_from_featuremap, resnet)

    return resnet


def adjust_last_layers(method, ft_from_featuremap, model):
    """
    Adjust the last layer from the network. This is only required
    when fine-tuning from feature map instead from the latent representation.

    """
    if method in ['simclr', 'swav', 'byol'] and ft_from_featuremap:
        model.avgpool = Identity()
        
    return model

def get_data_transforms(method):
    """
    Get proper data augmentation for the method given as parameter
    
    """
    
    #mean and std for imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]
    
    normalize = transforms.Normalize(mean=mean, std=std)
        
    train_trans = [transforms.RandomHorizontalFlip(),
                   transforms.RandomVerticalFlip(),
                   transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
                   transforms.RandomRotation(45),
                   transforms.ColorJitter(hue=0.2),
                   transforms.ToTensor()]
    
    val_trans =  [transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(),
                  transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
                  transforms.RandomRotation(45),
                  transforms.ColorJitter(hue=0.2),
                  transforms.ToTensor()]
    
    test_trans = [transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(),
                  transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
                  transforms.RandomRotation(45),
                  transforms.ColorJitter(hue=0.2),
                  transforms.ToTensor()]
    
    #simclr is the only one which was trained without imagenet normalization
    if method not in ['simclr']:
        train_trans.append(normalize)
        val_trans.append(normalize)
        test_trans.append(normalize)
    
    #dict to store the transformations for each step
    data_transforms = {
        'train': transforms.Compose(train_trans),
        'val': transforms.Compose(val_trans),
        'test': transforms.Compose(test_trans)
    }
    
    return data_transforms


def load_moco(path):
    #piece of code adapted from https://github.com/facebookresearch/moco/blob/master/main_lincls.py#L155-L177
    model = models.resnet50(pretrained=False)
    
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        msg = f"=> loaded pre-trained model '{path}'"
    else:
        msg = f"=> no checkpoint found at '{path}'"
    
    return model, msg
    


