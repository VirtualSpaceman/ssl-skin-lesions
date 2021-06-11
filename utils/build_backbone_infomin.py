import torch
import torch.nn as nn
from utils.resnet_infomin import model_dict
import torch.nn.functional as F
from collections import OrderedDict


class RGBSingleHead(nn.Module):
    """RGB model with a single linear/mlp projection head"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBSingleHead, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width

        self.encoder = model_dict[name](width=width)

        if head == 'linear':
            self.head = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 4
        elif name.endswith('x2'):
            return name[:-2], 2
        else:
            return name, 1

    def forward(self, x, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        feat = self.encoder(x)
        if mode == 0 or mode == 1:
            feat = self.head(feat)
        return feat


class RGBMultiHeads(RGBSingleHead):
    """RGB model with Multiple linear/mlp projection heads"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(RGBMultiHeads, self).__init__(name, head, feat_dim)

        self.head_jig = JigsawHead(dim_in=int(2048*self.width),
                                   dim_out=feat_dim,
                                   head=head)

    def forward(self, x, x_jig=None, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        if mode == 0:
            feat = self.head(self.encoder(x))
            feat_jig = self.head_jig(self.encoder(x_jig))
            return feat, feat_jig
        elif mode == 1:
            feat = self.head(self.encoder(x))
            return feat
        else:
            feat = self.encoder(x)
            return feat


class CMCSingleHead(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(CMCSingleHead, self).__init__()

        name, width = self._parse_width(name)
        dim_in = int(2048 * width)
        self.width = width

        self.encoder1 = model_dict[name](width=width, in_channel=1)
        self.encoder2 = model_dict[name](width=width, in_channel=2)

        if head == 'linear':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                Normalize(2)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    @staticmethod
    def _parse_width(name):
        if name.endswith('x4'):
            return name[:-2], 2
        elif name.endswith('x2'):
            return name[:-2], 1
        else:
            return name, 0.5

    def forward(self, x, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x1, x2 = torch.split(x, [1, 2], dim=1)
        feat1 = self.encoder1(x1)
        feat2 = self.encoder2(x2)
        if mode == 0 or mode == 1:
            feat1 = self.head1(feat1)
            feat2 = self.head2(feat2)
        return torch.cat((feat1, feat2), dim=1)


class CMCMultiHeads(CMCSingleHead):
    """CMC model with Multiple linear/mlp projection heads"""
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        super(CMCMultiHeads, self).__init__(name, head, feat_dim)

        self.head1_jig = JigsawHead(dim_in=int(2048*self.width),
                                    dim_out=feat_dim,
                                    head=head)
        self.head2_jig = JigsawHead(dim_in=int(2048*self.width),
                                    dim_out=feat_dim,
                                    head=head)

    def forward(self, x, x_jig=None, mode=0):
        # mode --
        # 0: normal encoder,
        # 1: momentum encoder,
        # 2: testing mode
        x1, x2 = torch.split(x, [1, 2], dim=1)
        feat1 = self.encoder1(x1)
        feat2 = self.encoder2(x2)

        if mode == 0:
            x1_jig, x2_jig = torch.split(x_jig, [1, 2], dim=1)
            feat1_jig = self.encoder1(x1_jig)
            feat2_jig = self.encoder2(x2_jig)

            feat1, feat2 = self.head1(feat1), self.head2(feat2)
            feat1_jig = self.head1_jig(feat1_jig)
            feat2_jig = self.head2_jig(feat2_jig)
            feat = torch.cat((feat1, feat2), dim=1)
            feat_jig = torch.cat((feat1_jig, feat2_jig), dim=1)
            return feat, feat_jig
        elif mode == 1:
            feat1, feat2 = self.head1(feat1), self.head2(feat2)
            return torch.cat((feat1, feat2), dim=1)
        else:
            return torch.cat((feat1, feat2), dim=1)


class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)


class JigsawHead(nn.Module):
    """Jigswa + linear + l2norm"""
    def __init__(self, dim_in, dim_out, k=9, head='linear'):
        super(JigsawHead, self).__init__()

        if head == 'linear':
            self.fc1 = nn.Linear(dim_in, dim_out)
        elif head == 'mlp':
            self.fc1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_out),
            )
        else:
            raise NotImplementedError('JigSaw head not supported: {}'.format(head))
        self.fc2 = nn.Linear(dim_out * k, dim_out)
        self.l2norm = Normalize(2)
        self.k = k

    def forward(self, x):
        bsz = x.shape[0]
        x = self.fc1(x)
        # ==== shuffle ====
        # this step can be moved to data processing step
        shuffle_ids = self.get_shuffle_ids(bsz)
        x = x[shuffle_ids]
        # ==== shuffle ====
        n_img = int(bsz / self.k)
        x = x.view(n_img, -1)
        x = self.fc2(x)
        x = self.l2norm(x)
        return x

    def get_shuffle_ids(self, bsz):
        n_img = int(bsz / self.k)
        rnd_ids = [torch.randperm(self.k) for i in range(n_img)]
        rnd_ids = torch.cat(rnd_ids, dim=0)
        base_ids = torch.arange(bsz)
        base_ids = torch.div(base_ids, self.k).long()
        base_ids = base_ids * self.k
        shuffle_ids = rnd_ids + base_ids
        return shuffle_ids

#default settings taken from https://github.com/HobbitLong/PyContrast/tree/master/pycontrast
OPT = {'method': 'InfoMin', 
        'modal': 'RGB', 
        'jigsaw': True, 
        'mem': 'moco', 
        'arch': 'resnet50', 
        'feat_dim': 128, 
        'head': 'mlp', 
        'ckpt': '/experimentos/pesos/infomin/InfoMin_800.pth', #custom path
        'aug_linear': 'NULL', 
        'n_class': 1000, 
        'aug': 'D'}
        
NAME_TO_FUNC = {
    'RGBSin': RGBSingleHead,
    'RGBMul': RGBMultiHeads,
    'CMCSin': CMCSingleHead,
    'CMCMul': CMCMultiHeads,
}


def load_encoder_weights(model):
        """load pre-trained weights for encoder

        Args:
          model: pretrained encoder, should be frozen
        """
        msg = "Empty Message"
        if OPT['ckpt']:
            ckpt = torch.load(OPT['ckpt'], map_location='cpu')
            state_dict = ckpt['model']
            if OPT['modal'] == 'RGB':
                # Unimodal (RGB) case
                encoder_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    if 'encoder' in k:
                        k = k.replace('encoder.', '')
                        encoder_state_dict[k] = v
                msg = model.encoder.load_state_dict(encoder_state_dict)
            else:
                # Multimodal (CMC) case
                encoder1_state_dict = OrderedDict()
                encoder2_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    if 'encoder1' in k:
                        k = k.replace('encoder1.', '')
                        encoder1_state_dict[k] = v
                    if 'encoder2' in k:
                        k = k.replace('encoder2.', '')
                        encoder2_state_dict[k] = v
                msg = model.encoder1.load_state_dict(encoder1_state_dict)
                msg += " " + model.encoder2.load_state_dict(encoder2_state_dict)
            print('Pre-trained weights loaded!', msg)
        else:
            print('==============================')
            print('warning: no pre-trained model!')
            print('==============================')
            msg = "warning: no pre-trained model!" 

        return model, msg

def build_model():
    # specify modal key
    branch = 'Mul' if OPT['jigsaw'] else 'Sin'
    model_key = OPT['modal'] + branch

    model = NAME_TO_FUNC[model_key](OPT['arch'], OPT['head'], OPT['feat_dim'])
    if OPT['mem'] == 'moco':
        model_ema = NAME_TO_FUNC[model_key](OPT['arch'], OPT['head'], OPT['feat_dim'])
    else:
        model_ema = None

    return model, model_ema




if __name__ == '__main__':
    model, _ = build_model()
    model, msg = load_encoder_weights(model)
    print(msg)



