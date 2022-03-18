# Model file Author Xiangyu Meng
import torch.nn as nn
import torch
import torch.fft as fft
import torch.nn.functional as F
from torch.nn.modules import activation
from torch.nn.modules.activation import Tanh
from torch.nn.modules.linear import Linear
from layers import *
import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from layers import *

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
class Discriminator(nn.Module):
    def __init__(self,seq_length,sample_length):
        super(Discriminator,self).__init__()
        self.main=nn.Sequential(
            nn.Linear(seq_length,sample_length),
            # nn.LeakyReLU(0.2,inplace=True),
            Mish(),

            # nn.Linear(sample_length,int(sample_length/2)),

            nn.Linear(sample_length,sample_length//2),
            # nn.LeakyReLU(0.2,inplace=True),
            Mish(),

            # nn.Linear(int(sample_length/2),code_dim)
            nn.Linear(sample_length//2,sample_length//4),
            Mish(),

            nn.Linear(sample_length//4,1)
            

        )
        # print(self.modules())

        self._init_weight()
    def forward(self,x):
        return self.main(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):

                nn.init.xavier_normal_(m.weight)

class VAE(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,dropout=True):
        super(VAE,self).__init__()
        
        model1=[
            nn.Linear(seq_length,sample_length),
            # nn.BatchNorm1d(sample_length),
            # nn.LeakyReLU(0.2,inplace=True),
            # Mish()
            nn.Tanh(),
            # nn.Dropout(0.5),
        ]
        if dropout:
            model1.append(nn.Dropout(0.5))
        self.downsample1=nn.Sequential(*model1)
        
        # self.attention=Attention(sample_length)

        self.encode_u=nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            
        )

        self.encode_si = nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            

        )

        self.decode = nn.Sequential(
			nn.Linear(code_dim, sample_length),
			nn.Tanh(),
            nn.Dropout(0.5),
			nn.Linear(sample_length, seq_length)
		)

        self._init_weight()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        assert not torch.isnan(std).any() and not torch.isnan(eps).any()
        return eps.mul(std).add_(mu)

    def dimention_reduction(self, x):
        h = self.downsample1(x)
        mu = self.encode_u(h)
        return mu
        
    def forward(self,x):
        x=self.downsample1(x)
        mu = self.encode_u(x)
        var = self.encode_si(x)
        z = self._reparameterize(mu,var)
        rec = self.decode(z)
        # x=self.attention(x)
        return rec, mu, var
    
    def _init_weight(self):
        # for m in self.downsample1.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # for m in self.downsample2.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)


class AVAE(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,dropout=True):
        super(AVAE,self).__init__()
        
        model1=[
            nn.Linear(seq_length,sample_length),
            # nn.BatchNorm1d(sample_length),
            # nn.LeakyReLU(0.2,inplace=True),
            # Mish()
            nn.Tanh(),
            
        ]
        if dropout:
            model1.append(nn.Dropout(0.5))
        self.downsample1=nn.Sequential(*model1)
        
        self.attention = nn.Sequential(
            nn.Linear(sample_length,sample_length),
            nn.Sigmoid(),
            nn.Dropout(0.2)
            
        )
        # self.attention=Attention(sample_length)

        self.encode_u=nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            nn.Dropout(0.5)
            
        )

        self.encode_si = nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            nn.Dropout(0.5)
            

        )

        self.decode = nn.Sequential(
			nn.Linear(code_dim, sample_length),
			nn.Tanh(),
            nn.Dropout(0.5),
			nn.Linear(sample_length, seq_length)
		)

        self._init_weight()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        assert not torch.isnan(std).any() and not torch.isnan(eps).any()
        return eps.mul(std).add_(mu)

    def dimention_reduction(self, x):
        h = self.downsample1(x)
        atten = self.attention(h)
        h = h*atten
        mu = self.encode_u(h)
        return mu
        
    def forward(self,x):
        x=self.downsample1(x)
        atten = self.attention(x)
        x = x*atten
        mu = self.encode_u(x)
        var = self.encode_si(x)
        z = self._reparameterize(mu,var)
        rec = self.decode(z)
        # x=self.attention(x)
        return rec, mu, var
    
    def _init_weight(self):
        # for m in self.downsample1.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # for m in self.downsample2.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)


class SAVAE(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,dropout=True):
        super(SAVAE,self).__init__()
        
        model1=[
            nn.Linear(seq_length,sample_length),
            # nn.BatchNorm1d(sample_length),
            # nn.LeakyReLU(0.2,inplace=True),
            # Mish()
            nn.Tanh(),
            
        ]
        if dropout:
            model1.append(nn.Dropout(0.5))
        self.downsample1=nn.Sequential(*model1)
        
        self.attention = Attention(sample_length)
        # self.attention=Attention(sample_length)

        self.encode_u=nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            nn.Dropout(0.5)
            
        )

        self.encode_si = nn.Sequential(
            nn.Linear(sample_length,code_dim),
            # nn.BatchNorm1d(code_dim),
            # nn.LeakyReLU(0.2,inplace=True)
            
            nn.Tanh(),
            nn.Dropout(0.5)
            

        )

        self.decode = nn.Sequential(
			nn.Linear(code_dim, sample_length),
			nn.Tanh(),
            nn.Dropout(0.5),
			nn.Linear(sample_length, seq_length)
		)

        self._init_weight()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        assert not torch.isnan(std).any() and not torch.isnan(eps).any()
        return eps.mul(std).add_(mu)

    def dimention_reduction(self, x):
        h = self.downsample1(x)
        h = self.attention(h)
        mu = self.encode_u(h)
        return mu
        
    def forward(self,x):
        x=self.downsample1(x)
        x = self.attention(x)
        
        mu = self.encode_u(x)
        var = self.encode_si(x)
        z = self._reparameterize(mu,var)
        rec = self.decode(z)
        # x=self.attention(x)
        return rec, mu, var
    
    def _init_weight(self):
        # for m in self.downsample1.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # for m in self.downsample2.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)


class Coxnnet(nn.Module):
    def __init__(self, nfeat):
        super(Coxnnet, self).__init__()
        self.fc1 = nn.Linear(nfeat, int(np.ceil(nfeat ** 0.5)))
        self.dropout=nn.Dropout(0.5)
        self.fc2 = nn.Linear(int(np.ceil(nfeat ** 0.5)), 1)
        self.init_hidden()

    def forward(self, x, coo=None):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def init_hidden(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

class CoxClassifierRNAseq(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(CoxClassifierRNAseq,self).__init__()
        self.freeze=freeze
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_300'))
        elif encoder_type =='attention':
            self.encoder=AVAE(seq_length,sample_length,code_dim)
            self.encoder.load_state_dict(torch.load('saved_models/VAEpretrain/g_ae_300'))
        # self.encoder=Encoder(seq_length,sample_length,code_dim)
        # self.encoder_mirna=Encoder(mi_seq_length,mi_sample_length,mi_code_dim)
        # self.classifier=nn.Sequential(
        #     nn.Linear(code_dim,int(np.ceil(code_dim ** 0.5))),
        #     nn.Tanh(),
        #     nn.Linear(int(np.ceil(code_dim ** 0.5)),label_dim),
        # )
        # self.cox=Coxnnet(code_dim)
        self.cox=nn.Sequential(nn.Linear(code_dim,1))
        # if rna_seq_dict is not None :
        #     self.transfer=True
        #     self.encoder.load_state_dict(torch.load(rna_seq_dict))
            # self.encoder_mirna.load_state_dict(torch.load(mirna_seq_dict))
            # self.decoder.load_state_dict(torch.load(state_dict_path))
        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        rna_code=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)

        return self.cox(rna_code),rna_code
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)




class CoxClassifierSRNAseq(nn.Module):
    def __init__(self,seq_length,sample_length,code_dim,encoder_type='basic',freeze=False):
        super(CoxClassifierSRNAseq,self).__init__()
        self.freeze=freeze
        if encoder_type == 'basic':
            self.encoder=VAE(seq_length,sample_length,code_dim)
            self.encoder.load_state_dict(torch.load('saved_models/NVAEpretrain/g_a_300'))
        elif encoder_type =='attention':
            self.encoder=SAVAE(seq_length,sample_length,code_dim)
            self.encoder.load_state_dict(torch.load('saved_models/SVAEpretrain/g_a_200'))
        # self.encoder=Encoder(seq_length,sample_length,code_dim)
        # self.encoder_mirna=Encoder(mi_seq_length,mi_sample_length,mi_code_dim)
        # self.classifier=nn.Sequential(
        #     nn.Linear(code_dim,int(np.ceil(code_dim ** 0.5))),
        #     nn.Tanh(),
        #     nn.Linear(int(np.ceil(code_dim ** 0.5)),label_dim),
        # )
        # self.cox=Coxnnet(code_dim)
        self.cox=nn.Sequential(nn.Linear(code_dim,1))
        # if rna_seq_dict is not None :
        #     self.transfer=True
        #     self.encoder.load_state_dict(torch.load(rna_seq_dict))
            # self.encoder_mirna.load_state_dict(torch.load(mirna_seq_dict))
            # self.decoder.load_state_dict(torch.load(state_dict_path))
        
        if self.freeze==True:
            self.set_freeze_by_names(self,'ecoder',freeze=True)
    
    def forward(self,x_rna):
        rna_code=self.encoder.dimention_reduction(x_rna)
        # mirna_code=self.encoder_mirna(x_mirna)

        return self.cox(rna_code),rna_code
    
    def _init_weight(self):
        if self.transfer==True:
            for m in self.classifier.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        else:
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    nn.init.xavier_normal_(m.weight)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_cox(hazards, labels):
    # This accuracy is based on estimated survival events against true survival events
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    labels = labels.data.cpu().numpy()
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)
    
def CIndex(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]: concord = concord + 1
                    elif hazards[j] < hazards[i]: concord = concord + 0.5

    return(concord/total)
    
def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    hazards = hazards.cpu().numpy().reshape(-1)
    return(concordance_index(survtime_all, -hazards, labels))
        
def frobenius_norm_loss(a, b):
    loss = torch.sqrt(torch.sum(torch.abs(a-b)**2))
    return loss
