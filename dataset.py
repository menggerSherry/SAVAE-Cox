import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
# import torchvision.transforms as transforms
import h5py
import random

# device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class PanDataset(Dataset):
    def __init__(self, root, mode='train', transforms_=None):
        # self.transforms = transforms.Compose(transforms_)
        self.path=os.path.join(root,'GDC_PANCANCER.htseq_fpkm-uq_final.hdf5')
        self.mode=mode
      
    # def norm(self,x):
    #     out=(x-0.5)/0.5
    #     return out


    def __getitem__(self, index):
        # data=pd.read_csv(self.path,sep='\t',index_col=0).reset_index(drop=True)
        data=h5py.File(self.path,'r')
        g=data['pancancer_exp']
        # print(g.keys())
        # d=g["test1"]
        # print(d.name())
        exp_data=g['%s_%d'%(self.mode,index)][:]
        # exp_data=self.norm(exp_data)
        exp_data=torch.from_numpy(exp_data)
        # (-1,1)
        # exp_data=self.norm(exp_data)
        target=exp_data.clone()
        data.close()
        return {'exp':exp_data,'target':target}

    def __len__(self):
        data=h5py.File(self.path,'r')
        g=data['dataset_dim']
        length=g['%s'%self.mode][0]
        return length


class PanMiRNADataset(Dataset):
    def __init__(self, root, mode='train', transforms_=None):
        # self.transforms = transforms.Compose(transforms_)
        self.path=os.path.join(root,'GDC_PANCANCER.mirna_final.hdf5')
        self.mode=mode
      
    # def norm(self,x):
    #     out=(x-0.5)/0.5
    #     return out


    def __getitem__(self, index):
        # data=pd.read_csv(self.path,sep='\t',index_col=0).reset_index(drop=True)
        data=h5py.File(self.path,'r')
        g=data['pancancer_exp']
        # print(g.keys())
        # d=g["test1"]
        # print(d.name())
        exp_data=g['%s_%d'%(self.mode,index)][:]
        # exp_data=self.norm(exp_data)
        exp_data=torch.from_numpy(exp_data)
        # (-1,1)
        # exp_data=self.norm(exp_data)
        target=exp_data.clone()
        data.close()
        return {'exp':exp_data,'target':target}

    def __len__(self):
        data=h5py.File(self.path,'r')
        g=data['dataset_dim']
        length=g['%s'%self.mode][0]
        return length

def get_miloader(root, batch_size,mode, num_workers=2,shuffle=True,drop_last=False):
    # if dataset_type == 'pan':
    #     dataset = PanDataset(root, mode )
    # elif dataset_type == 'cox':
    #     dataset = CoxDataset(root,omics_type,kf,mode)
    dataset = PanMiRNADataset(root,mode)
    # print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

    
class CoxDataset(Dataset):

    def __init__(self,root,omics_type,kf,mode='train'):
        # if mode != 'train' or mode != 'test':
        #      raise NotImplementedError("error mode type")

        path=os.path.join(root,'%s.5_folds.hdf5'%omics_type)
        data_file=h5py.File(path,'r')
        data_group=data_file['exp']
        fold_group=data_group['cross_%d'%kf]
        self.data=fold_group[mode][:]
        data_file.close()
        self.data=torch.from_numpy(self.data)
        print(self.data.shape)
    
    def __getitem__(self,index):
        
        exp_data=self.data[index,:-1287]
        mirna_exp_data=self.data[index,-1287:-2]
        os_event=self.data[index,-2]
        os_time=self.data[index,-1]
        return {'exp':exp_data,'mi_exp':mirna_exp_data,'event':os_event,'time':os_time}
    def __len__(self):
        return self.data.shape[0]

# data=CoxDataset('data_new','TCGA-BLCA',1)
# x=data[1]
# print(x['exp'].size())
# print(x['exp'].dtype)
# print(x['mi_exp'].size())
# print(x['mi_exp'].dtype)


def get_loader(root, batch_size,mode, num_workers=6,kf=None,omics_type=None,dataset_type='pan',shuffle=True,drop_last=False):
    if dataset_type == 'pan':
        dataset = PanDataset(root, mode )
    elif dataset_type == 'cox':
        dataset = CoxDataset(root,omics_type,kf,mode)
    # print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

