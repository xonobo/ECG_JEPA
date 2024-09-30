import torch
import torch.nn as nn
import torch.nn.functional as F
from cpc.cpc import *
# from time_jepa_ver4_pos_once import Time_jepa
from time_jepa_ver4 import Time_jepa
# from time_jepa_ver3 import Time_jepa

class CMSC(nn.Module):
    
    """ CNN for Self-Supervision """
    
    def __init__(self, p1=0.1,p2=0.1,p3=0.1,embedding_dim=192,trial='',device=''):
        super(CMSC,self).__init__()
        
        self.c1 = 1 #b/c single time-series
        self.c2 = 4 #4
        self.c3 = 16 #16
        self.c4 = 32 #32
        self.k=7 #self.kernel size #7 
        self.s=3 #stride #3

        self.embedding_dim = embedding_dim
    
        self.dropout1 = nn.Dropout(p=p1) #0.2 drops pixels following a Bernoulli
        self.dropout2 = nn.Dropout(p=p2) #0.2
        self.dropout3 = nn.Dropout(p=p3)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool1d(2)
        self.trial = trial
        self.device = device
        
        self.view_modules = nn.ModuleList()
        self.view_linear_modules = nn.ModuleList()
        self.view_modules.append(nn.Sequential(
        nn.Conv1d(self.c1,self.c2,self.k,self.s, padding=1),
        nn.BatchNorm1d(self.c2),
        nn.ReLU(),
        nn.MaxPool1d(2),
        self.dropout1,
        nn.Conv1d(self.c2,self.c3,self.k,self.s),
        nn.BatchNorm1d(self.c3),
        nn.ReLU(),
        nn.MaxPool1d(2),
        self.dropout2,
        nn.Conv1d(self.c3,self.c4,self.k,self.s),
        nn.BatchNorm1d(self.c4),
        nn.ReLU(),
        # nn.MaxPool1d(2),
        self.dropout3
        ))
        self.view_linear_modules.append(nn.Linear(self.c4*10,self.embedding_dim))

    def contrastive_batch(self,x):
        """ Forward Pass on Batch of Inputs 
        Args:
            x (torch.Tensor): (bs, c, T)
        Outputs:
            h (torch.Tensor): latent embedding for each of the N views (BxHxN)
        """
        bs,c,T = x.shape
        x1 = x[:,:,:T//2] #(bs, c, T//2)
        x2 = x[:,:,T//2:] #(bs, c, T//2)

        x1 = x1.reshape(-1,T//2).unsqueeze(1)
        x2 = x2.reshape(-1,T//2).unsqueeze(1)
        
        h1 = self.view_modules[0](x1) # (bs*c, self.c4, 10)
        h2 = self.view_modules[0](x2) # (bs*c, self.c4, 10)

        h1 = h1.reshape(-1,self.c4*10) # (bs*c self.c4*10)
        h2 = h2.reshape(-1,self.c4*10)

        h1 = self.view_linear_modules[0](h1) # (bs*c, embedding_dim)
        h2 = self.view_linear_modules[0](h2) # (bs*c, embedding_dim)

        h1 = h1.reshape(bs,c,self.embedding_dim) # (bs, c, embedding_dim)
        h2 = h2.reshape(bs,c,self.embedding_dim) # (bs, c, embedding_dim)

        return h1,h2
    
    def representation(self,x):
        h1, h2 = self.contrastive_batch(x) 
        h = torch.cat([h1,h2],dim=2) # (bs, c, 2*embedding_dim)
        h = torch.mean(h,dim=1) # (bs, 2*embedding_dim)
        return h


    def forward(self,x):
        bs, c , T = x.shape
        h1, h2 = self.contrastive_batch(x)

        loss = 0.
        for i in range(c):
            # loss += infoNCE(h1[:,i],h2[:,i])
            loss += self._loss(h1[:,i],h2[:,i])

        return loss

    def _loss(self,x, y, temperature=0.1):
        """
        Computes the InfoNCE loss between embeddings x and y.
        
        Args:
        x (torch.Tensor): Tensor of shape (bs, dim)
        y (torch.Tensor): Tensor of shape (bs, dim)
        temperature (float): Temperature scaling parameter
        
        Returns:
        torch.Tensor: Scalar loss value
        """
        # Ensure the embeddings are normalized
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        
        # Compute the similarity matrix
        similarity_matrix = torch.matmul(x, y.T) / temperature
        
        # Create labels (0 to bs-1)
        bs = x.shape[0]
        labels = torch.arange(bs).to(x.device)
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss



__all__ = ['CPCEncoder', 'CPCModel']

# Cell
import torch
import torch.nn.functional as F
import torch.nn as nn
from cpc.basic_conv1d import _conv1d
import numpy as np

# from .basic_conv1d import listify, bn_drop_lin

# Cell
class CPCEncoder(nn.Sequential):
    'CPC Encoder'

    # strides = [1, 1, 1, 1]
    # kss = [1, 1, 1, 1]
    # features = [512, 512, 512, 512]
    def __init__(self, input_channels, strides=[5,4,2,2,2], kss=[10,8,4,4,4], features=[512,512,512,512],bn=False):
        assert(len(strides)==len(kss) and len(strides)==len(features))
        lst = []
        for i,(s,k,f) in enumerate(zip(strides,kss,features)):
            lst.append(_conv1d(input_channels if i==0 else features[i-1],f,kernel_size=k,stride=s,bn=bn))
        super().__init__(*lst)
        self.downsampling_factor = np.prod(strides)
        self.output_dim = features[-1]
        # output: bs, output_dim, seq//downsampling_factor
    def encode(self, input):
        #bs = input.size()[0]
        #ch = input.size()[1]
        #seq = input.size()[2]
        #segments = seq//self.downsampling_factor
        #input_encoded = self.forward(input[:,:,:segments*self.downsampling_factor]).transpose(1,2) #bs, seq//downsampling, encoder_output_dim (standard ordering for batch_first RNNs)
        input_encoded = self.forward(input)
        input_encoded = input_encoded.transpose(1,2)
        return input_encoded

# Cell
class CPCModel(nn.Module):
    "CPC model"
    def __init__(self, input_channels, strides=[5,4,2,2,2], kss=[10,8,4,4,4], features=[512,512,512,512],bn_encoder=False, n_hidden=512,n_layers=2,mlp=False,lstm=True,bias_proj=False, num_classes=None, concat_pooling=True, ps_head=0.5,lin_ftrs_head=[512],bn_head=True,skip_encoder=False):
        super().__init__()
        assert(skip_encoder is False or num_classes is not None)#pretraining only with encoder
        self.encoder = CPCEncoder(input_channels,strides=strides,kss=kss,features=features,bn=bn_encoder) if skip_encoder is False else None
        self.encoder_output_dim = self.encoder.output_dim if skip_encoder is False else None
        self.encoder_downsampling_factor = self.encoder.downsampling_factor if skip_encoder is False else None
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.mlp = mlp

        self.num_classes = num_classes
        self.concat_pooling = concat_pooling

        self.rnn = nn.LSTM(self.encoder_output_dim if skip_encoder is False else input_channels,n_hidden,num_layers=n_layers,batch_first=True) if lstm is True else nn.GRU(self.encoder.output_dim,n_hidden,num_layers=n_layers,batch_first=True)
        self.concat_pooling_ = AdaptiveConcatPoolRNN() if concat_pooling is True else None

        if(num_classes is None): #pretraining
            if(mlp):# additional hidden layer as in simclr
                self.proj = nn.Sequential(nn.Linear(n_hidden, n_hidden),nn.ReLU(inplace=True),nn.Linear(n_hidden, self.encoder_output_dim,bias=bias_proj))
            else:
                self.proj = nn.Linear(n_hidden, self.encoder_output_dim,bias=bias_proj)

    def representation(self, input):
        if(self.encoder is not None):
            input_encoded = self.encoder.encode(input)
        else:
            input_encoded = input.transpose(1,2)

        output_rnn, _ = self.rnn(input_encoded)

        return self.concat_pooling_(output_rnn) if self.concat_pooling else output_rnn

    def forward(self, input):
        # input shape bs,ch,seq
        if(self.encoder is not None):
            input_encoded = self.encoder.encode(input)
        else:
            input_encoded = input.transpose(1,2) #bs, seq, channels
        output_rnn, _ = self.rnn(input_encoded) #output_rnn: bs, seq, n_hidden
        if(self.num_classes is None):#pretraining
            return input_encoded, self.proj(output_rnn)
        else:#classifier
            output = output_rnn.transpose(1,2)#bs,n_hidden,seq (i.e. standard CNN channel ordering)
            if(self.concat_pooling is False):
                output = output[:,:,-1]
            return self.head(output)

    def get_layer_groups(self):
        return (self.encoder,self.rnn,self.head)

    def get_output_layer(self):
        return self.head[-1]

    def set_output_layer(self,x):
        self.head[-1] = x

    # step_predicted=12
    def cpc_loss(self,input, target=None, steps_predicted=5, n_false_negatives=9, negatives_from_same_seq_only=False, eval_acc=False):
        assert(self.num_classes is None)

        input_encoded, output = self.forward(input) #input_encoded: bs, seq, features; output: bs,seq,features
        input_encoded_flat = input_encoded.reshape(-1,input_encoded.size(2)) #for negatives below: -1, features

        bs = input_encoded.size()[0]
        seq = input_encoded.size()[1]

        loss = torch.tensor(0,dtype=torch.float32).to(input.device)
        tp_cnt = torch.tensor(0,dtype=torch.int64).to(input.device)
        # steps_predicted = 12
        for i in range(input_encoded.size()[1]-steps_predicted):
            # input_incoded.size() = (bs, seq, channels(vectors))
            positives = input_encoded[:,i+steps_predicted].unsqueeze(1) #bs,1,encoder_output_dim
            if(negatives_from_same_seq_only): # True
                # seq = 1000
                # n_false_negatives = 128
                idxs = torch.randint(0,(seq-1),(bs*n_false_negatives,)).to(input.device)
            else:#negative from everywhere
                idxs = torch.randint(0,bs*(seq-1),(bs*n_false_negatives,)).to(input.device)
            # print(f"{idxs=}")
            # print(f"{idxs.size()=}")
            idxs_seq = torch.remainder(idxs,seq-1) #bs*false_neg
            # print(f"{idxs_seq=}")
            idxs_seq2 = idxs_seq * (idxs_seq<(i+steps_predicted)).long() +(idxs_seq+1)*(idxs_seq>=(i+steps_predicted)).long()#bs*false_neg
            # print(f"{idxs_seq2=}")
            if(negatives_from_same_seq_only):
                idxs_batch = torch.arange(0,bs).repeat_interleave(n_false_negatives).to(input.device)
                # print(f"{idxs_batch=}")
                # print(f"{idxs_batch.size()=}")
            else:
                idxs_batch = idxs//(seq-1)
            idxs2_flat = idxs_batch*seq+idxs_seq2 #for negatives from everywhere: this skips step i+steps_predicted from the other sequences as well for simplicity

            negatives = input_encoded_flat[idxs2_flat].view(bs,n_false_negatives,-1) #bs*false_neg, encoder_output_dim
            candidates = torch.cat([positives,negatives],dim=1)#bs,1+false_neg,encoder_output_dim

            preds=torch.sum(output[:,i].unsqueeze(1)*candidates,dim=-1) #bs,(1+false_neg) # sum 은 global average pooling 같은 느낌인가?

            targs = torch.zeros(bs, dtype=torch.int64).to(input.device) # 0번째 class가 true이기 때문에 !!!!!! target이 0 !!

            if(eval_acc):
                preds_argmax = torch.argmax(preds,dim=-1)
                tp_cnt += torch.sum(preds_argmax == targs)

            loss += F.cross_entropy(preds,targs)
        if(eval_acc):
            return loss, tp_cnt.float()/bs/(input_encoded.size()[1]-steps_predicted)
        else:
            return loss

#copied from RNN1d
class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
    def forward(self,x):
        #input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        if(self.bidirectional is False):
            t3 = x[:,:,-1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:,:channels,-1],x[:,channels:,0]],1)
        out=torch.cat([t1.squeeze(-1),t2.squeeze(-1),t3],1) #output shape bs, 3*ch
        return out

import torch.nn.functional as F
import numpy as np


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

def ResNet50_1D():
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3])

# Projection Head for SimCLR
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# SimCLR model integrating the ResNet-50
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLR, self).__init__()
        self.encoder = base_model
        self.projector = ProjectionHead(2048, out_dim)

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        #print(h_i.shape, self.n_features)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
    
    def representation(self, x):
        return self.encoder(x)
    
from st_mem import ST_MEM_ViT
    

def load_encoder(model_name, ckpt_dir, leads=None):
    model_names = ['cpc', 'cmsc', 'simclr', 'ejepa_random', 'ejepa_multiblock','st_mem']
    
    assert model_name in model_names, f"Model name must be one of {model_names}"
    
    if leads is not None:
        assert model_name in ['ejepa_random', 'ejepa_multiblock', 'st_mem'], f"Model {model_name} does not support reduced leads"

    if leads is None:
        leads = [0,1,2,3,4,5,6,7]

    if model_name == 'cpc':
        hparams = {
            "accumulate":                   1,
            "batch_size":                   32,
            "bias":                         False,
            "data":                         ['./ecg_data_processed/ribeiro_fs100'],
            "discriminative_lr_factor":     0.1,
            "distributed_backend":          None,
            "dropout_head":                 0.5,
            "epochs":                       1000,
            "executable":                   "cpc",
            "fc_encoder":                   True,
            "finetune":                     False,
            "finetune_dataset":             "thew",
            "gpus":                         1,
            "gru":                          False,
            "input_channels":               8,
            "input_size":                   1000,
            "lin_ftrs_head":                [512],
            "linear_eval":                  False,
            "lr":                           0.0001,
            "lr_find":                      False,
            "metadata": None,
            "mlp":                          True,
            "n_false_negatives":            128,
            "n_hidden":                     512,
            "n_layers":                     2,
            "negatives_from_same_seq_only": True,
            "no_bn_encoder":                False,
            "no_bn_head":                   False,
            "normalize":                    True,
            "num_nodes":                    1,
            "optimizer":                    "adam",
            "output_path":                  "./runs/cpc/all",
            "precision":                    16,
            "pretrained": None,
            "resume": None,
            "skip_encoder":                 False,
            "steps_predicted":              12,
            "train_head_only":              False,
            "weight_decay":                 0.001,

        }

        encoder = CPCModel(
            input_channels=hparams['input_channels'], # 12
            strides=[1]*4, 
            kss=[1]*4, 
            features=[512]*4,
            n_hidden=hparams['n_hidden'], # 512
            n_layers=hparams['n_layers'], # 2
            mlp=hparams['mlp'], #
            lstm=not(hparams['gru']),
            bias_proj=hparams['bias'],
            num_classes=None,
            skip_encoder=hparams['skip_encoder'],
            bn_encoder=not(hparams['no_bn_encoder']),
            # lin_ftrs_head=[] if hparams.linear_eval else eval(hparams.lin_ftrs_head),
            lin_ftrs_head=[] if hparams['linear_eval'] else 512,
            ps_head=0 if hparams['linear_eval'] else hparams['dropout_head'],
            bn_head=False if hparams['linear_eval'] else not(hparams['no_bn_head'])
        )
        ckpt = torch.load(ckpt_dir)
        encoder.load_state_dict(ckpt['base_model'])
        embed_dim = 7500

    elif model_name == 'cmsc':
        encoder = CMSC()
        ckpt = torch.load(ckpt_dir)
        encoder.load_state_dict(ckpt['encoder'])
        embed_dim = 384

    elif model_name == 'ejepa_random':
        params = {
            'encoder_embed_dim': 768,
            'encoder_depth': 12,
            'encoder_num_heads': 16,
            'predictor_embed_dim': 384,
            'predictor_depth': 6,
            'predictor_num_heads': 12,
            'drop_path_rate': 0.,
            'c': 8,
            'pos_type': 'sincos',
            'mask_scale': (0, 0),
            'mask_type': 'random', # or block
            'leads': leads
        }

        encoder = Time_jepa(**params).encoder
        ckpt = torch.load(ckpt_dir)
        encoder.load_state_dict(ckpt['encoder'])
        embed_dim = 768

    elif model_name == 'ejepa_multiblock':
        params = {
            'encoder_embed_dim': 768,
            'encoder_depth': 12,
            'encoder_num_heads': 16,
            'predictor_embed_dim': 384,
            'predictor_depth': 6,
            'predictor_num_heads': 12,
            'drop_path_rate': 0.,
            'c': 8,
            'pos_type': 'sincos',
            'mask_scale': (0, 0),
            'mask_type': 'multiblock', # or block
            'leads': leads
        }

        encoder = Time_jepa(**params).encoder
        ckpt = torch.load(ckpt_dir)
        encoder.load_state_dict(ckpt['encoder'])
        embed_dim = 768

    elif model_name == 'simclr':
        base_model = ResNet50_1D()
        encoder = SimCLR(base_model=base_model, out_dim=128)
        ckpt = torch.load(ckpt_dir)
        encoder.load_state_dict(ckpt['model'])
        embed_dim = 2048

    elif model_name == 'st_mem':
        encoder = ST_MEM_ViT(seq_len=2250, patch_size=75, num_leads=12)
        ckpt = torch.load(ckpt_dir)
        encoder.load_state_dict(ckpt['model'])
        embed_dim = 768

    return encoder, embed_dim











# For linear model of CPC, use the following:
# hparams = {
#     "accumulate":                   1,
#     "batch_size":                   32,
#     "bias":                         False,
#     "data":                         ['./ecg_data_processed/ribeiro_fs100'],
#     "discriminative_lr_factor":     0.1,
#     "distributed_backend":          None,
#     "dropout_head":                 0.5,
#     "epochs":                       1000,
#     "executable":                   "cpc",
#     "fc_encoder":                   True,
#     "finetune":                     False,
#     "finetune_dataset":             "thew",
#     "gpus":                         1,
#     "gru":                          False,
#     "input_channels":               8,
#     "input_size":                   1000,
#     "lin_ftrs_head":                [512],
#     "linear_eval":                  False,
#     "lr":                           0.0001,
#     "lr_find":                      False,
#     "metadata": None,
#     "mlp":                          True,
#     "n_false_negatives":            128,
#     "n_hidden":                     512,
#     "n_layers":                     2,
#     "negatives_from_same_seq_only": True,
#     "no_bn_encoder":                False,
#     "no_bn_head":                   False,
#     "normalize":                    True,
#     "num_nodes":                    1,
#     "optimizer":                    "adam",
#     "output_path":                  "./runs/cpc/all",
#     "precision":                    16,
#     "pretrained": None,
#     "resume": None,
#     "skip_encoder":                 False,
#     "steps_predicted":              12,
#     "train_head_only":              False,
#     "weight_decay":                 0.001,

# }
# encoder = CPCModel(
#     input_channels=hparams.input_channels, # 12
#     strides=[1]*4, 
#     kss=[1]*4, 
#     features=[512]*4,
#     n_hidden=hparams.n_hidden, # 512
#     n_layers=hparams.n_layers, # 2
#     mlp=hparams.mlp, #
#     lstm=not(hparams.gru),
#     bias_proj=hparams.bias,
#     num_classes=None,
#     skip_encoder=hparams.skip_encoder,
#     bn_encoder=not(hparams.no_bn_encoder),
#     # lin_ftrs_head=[] if hparams.linear_eval else eval(hparams.lin_ftrs_head),
#     lin_ftrs_head=[] if hparams.linear_eval else 512,
#     ps_head=0 if hparams.linear_eval else hparams.dropout_head,
#     bn_head=False if hparams.linear_eval else not(hparams.no_bn_head)
# ).to(device)