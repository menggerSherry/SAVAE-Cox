# from matplotlib.font_manager import _Weight
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import os
from torch.utils.data import dataloader

from tqdm.std import TqdmMonitorWarning
sys.path.append("./model")
sys.path.append("./loss")
sys.path.append('./utils')
import dataset
import VGANCox
import argparse
from torch.backends import cudnn
import datetime
import time
from tqdm import tqdm
import numpy as np
import logging
from sklearn.metrics import r2_score
import summary
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import gc
import pickle


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark=True

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data_new', help='dataset path')
parser.add_argument('--input_g',type=int,default=100)
parser.add_argument('--mid_g',type=int,default=1024)
parser.add_argument('--mid_d',type=int,default=2048)
parser.add_argument('--out_d',type=int,default=100)
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--beta1',type=float,default=0.5)
parser.add_argument('--beta2',type=float,default=0.999)
parser.add_argument('--epoch', type=int, default=0, help='epoch')
parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')
parser.add_argument('--n_epochs', type=int, default=30, help='train epoch')
parser.add_argument('--checkpoints_interval', type=int, default=20, help='check')
parser.add_argument('--sample_interval',type=int,default=2)
parser.add_argument('--result_log',type=str,default='COX')
parser.add_argument('--model_path',type=str,default='COX')
parser.add_argument('--batch_size',type=int,default=512)
parser.add_argument('--seq_length',type=int,default=20034)
parser.add_argument('--sample_length',type=int,default=1024)
parser.add_argument('--latern_dim',type=int,default=256)
parser.add_argument('--n_critic',type=int,default=4)
parser.add_argument('--miseq_length',type=int,default=1285)
parser.add_argument('--misample_length',type=int,default=256)
parser.add_argument('--milatern_dim',type=int,default=100)
parser.add_argument('--rna_seq_dict',type=str,default='./saved_models/pretrain/g_e_200')
parser.add_argument('--mirna_seq_dict',type=str,default='./saved_models/pretrain/g_mirna_e_200')
parser.add_argument('--lasso',type=bool,default=True)
parser.add_argument('--encoder_type',type=str,default='attention')
parser.add_argument('--omics_type',type=str,default='TCGA-BRCA')

config = parser.parse_args()


def train(config,dataloader,dataloader_val,num_epochs, batch_size, learning_rate,  measure, verbose,save_state,time_str,idx):
    
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    model=VGANCox.CoxClassifierSRNAseq(config.seq_length,config.sample_length,config.latern_dim,config.encoder_type)
    model=model.cuda()
    summary.summary(model,(config.seq_length,),'gen_1')
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.1)
    c_index_list = {}
    c_index_list['train'] = []
    c_index_list['test'] = []
    loss_nn_all = []
    pvalue_all = []
    c_index_all = []
    acc_train_all = []
    c_index_train_best = 0
    code_output = None
    code_val=None
    model.train()
    max_pre=0
    best_log_p=0
    best_train_p_value = 0
    lbl_pred_train = None
    for epoch in tqdm(range(num_epochs)):
        
        
        lbl_pred_all = None
        lbl_all = None
        survtime_all = None
        code_final = None
        loss_nn_sum = 0
        iter = 0
        gc.collect()
        for i,data in enumerate(dataloader):
            optimizer.zero_grad()
            exp=data['exp'].cuda().to(torch.float32)
            exp_mirna=data['mi_exp'].cuda().to(torch.float32)
            # exp,exp=aug.aug(None,exp,exp,500,config.batch_size)
            survtime=data['time']
            lbl=data['event'].cuda()
            lbl_pred,code=model(exp)
            if iter == 0:
                lbl_pred_all = lbl_pred
                survtime_all = survtime
                lbl_all = lbl
                code_final = code
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                lbl_all = torch.cat([lbl_all, lbl])
                survtime_all = torch.cat([survtime_all, survtime])
                code_final = torch.cat([code_final, code])
            
            current_batch_len = len(survtime)
            R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
            for i in range(current_batch_len):
                for j in range(current_batch_len):
                    R_matrix_train[i,j] = survtime[j] >= survtime[i]
        
            train_R = torch.FloatTensor(R_matrix_train)
            train_R = train_R.cuda()
            train_ystatus = lbl
            theta = lbl_pred.reshape(-1)
            exp_theta = torch.exp(theta)
            
            loss = -torch.mean( (theta - torch.log(torch.sum( exp_theta*train_R ,dim=1))) * train_ystatus.float() )

            # if config.lasso:
            #     L1 = nn.L1Loss()
            #     model_param = torch.cat([x.view(-1) for x in model.parameters()])
            #     loss = loss.cuda() + L1(model_param.cuda(), torch.zeros(model_param.size()).cuda().float())
            loss.backward()
            optimizer.step()

            iter += 1
            torch.cuda.empty_cache()

        code_final_4_original_data = code_final.data.cpu().numpy()
        if measure or epoch == (num_epochs - 1):
            acc_train = VGANCox.accuracy_cox(lbl_pred_all.data, lbl_all)
            pvalue_pred = VGANCox.cox_log_rank(lbl_pred_all.data, lbl_all, survtime_all)
            c_index = VGANCox.CIndex_lifeline(lbl_pred_all.data, lbl_all, survtime_all)
            c_index_list['train'].append(c_index)
            # TRAIN
            if c_index > c_index_train_best:
                c_index_train_best = c_index
                code_output = code_final_4_original_data
                os_train= survtime_all
                lbl_pred_train=lbl_pred_all
                best_train_p_value=pvalue_pred
            if verbose:
                print('\n[Training]\t loss (nn):{:.4f}'.format(loss_nn_sum),
                      'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
            pvalue_all.append(pvalue_pred)
            c_index_all.append(c_index)
            loss_nn_all.append(loss_nn_sum)
            acc_train_all.append(acc_train)
            whichset = 'test'

            # VAL
            code_validation, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_test_all,lbl_pred_all,  OS = \
                test(model, dataloader_val, whichset,  batch_size,  verbose)

            if c_index_pred > max_pre:
                max_pre=c_index_pred
                best_log_p=pvalue_pred
                code_val=code_validation
                os_val=OS
                if save_state:
                    torch.save(model.state_dict(),'result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/model.pth')
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/c_index_list_by_epochs.pickle', 'wb') as handle:
                        pickle.dump(c_index_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/hazard_ratios_lbl_pred_all_train.pickle', 'wb') as handle:
                        pickle.dump(lbl_pred_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/OS_event_train.pickle', 'wb') as handle:
                        pickle.dump(lbl_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/OS_train.pickle', 'wb') as handle:
                        pickle.dump(os_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/code_train.pickle', 'wb') as handle:
                        pickle.dump(code_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/hazard_ratios_lbl_pred_all_test.pickle', 'wb') as handle:
                        pickle.dump(lbl_pred_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/OS_event_test.pickle', 'wb') as handle:
                        pickle.dump(lbl_test_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/OS_test.pickle', 'wb') as handle:
                        pickle.dump(OS, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/code_test.pickle', 'wb') as handle:
                        pickle.dump(code_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
            c_index_list['test'].append(c_index_pred)
    # torch.save(model.state_dict(),'result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, idx+1) + '/model.pth')       
    return(model, loss_nn_all, pvalue_all, c_index_all, c_index_list, acc_train_all, c_index_train_best, code_output,os_train, lbl_all, best_train_p_value,lbl_pred_train, max_pre,code_val,os_val,lbl_test_all, best_log_p,lbl_pred_all,OS)
            
def test(model, dataloader_val, whichset,  batch_size,  verbose):
    lbl_pred_all = None
    lbl_all = None
    survtime_all = None
    code_final = None
    loss_nn_sum = 0
    model.eval()
    iter = 0

    for data in dataloader_val:
        exp=data['exp'].cuda().to(torch.float32)
        mi_exp=data['mi_exp'].cuda().to(torch.float32)
        lbl=data['event'].cuda()
        survtime=data['time']
        lbl_pred,code=model(exp)
    
        if iter == 0:
                lbl_pred_all = lbl_pred
                lbl_all = lbl
                survtime_all = survtime
                code_final = code
        else:
            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
            lbl_all = torch.cat([lbl_all, lbl])
            survtime_all = torch.cat([survtime_all, survtime])
            code_final = torch.cat([code_final, code])
        
        current_batch_len = len(survtime)
        R_matrix_test = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_matrix_test[i,j] = survtime[j] >= survtime[i]

        test_R = torch.FloatTensor(R_matrix_test)
        test_R = Variable(test_R)
        test_R = test_R.cuda()

        test_ystatus = lbl
        theta = lbl_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_nn = -torch.mean( (theta - torch.log(torch.sum( exp_theta*test_R ,dim=1))) * test_ystatus.float() )
        loss_nn_sum = loss_nn_sum + loss_nn.data.item()
        iter += 1

    code_final_4_original_data = code_final.data.cpu().numpy()
    acc_test = VGANCox.accuracy_cox(lbl_pred_all.data, lbl_all)
    pvalue_pred = VGANCox.cox_log_rank(lbl_pred_all.data, lbl_all, survtime_all)
    c_index = VGANCox.CIndex_lifeline(lbl_pred_all.data, lbl_all, survtime_all)
    if verbose > 0:
        print('\n[{:s}]\t\tloss (nn):{:.4f}'.format(whichset, loss_nn_sum),
                      'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    return(code_final_4_original_data, loss_nn_sum, acc_test, \
           pvalue_pred, c_index, lbl_all,lbl_pred_all.data.cpu().numpy().reshape(-1), survtime_all)
            


def main():
    os.makedirs('result/COX_%s'%config.encoder_type,exist_ok=True)
    time_str  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
    # os.makedirs('result/COX/%s'%time_str,exist_ok=True)

    cudnn.benchmark = True
    # os.makedirs('saved_models/%s/' % (config.model_path), exist_ok=True)
    os.makedirs('result/COX_%s/%s_%s'%(config.encoder_type,time_str,config.omics_type),exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                    filename='result/COX_%s/%s_%s/%s.log'%(config.encoder_type,time_str,config.omics_type,config.result_log),
                    filemode='a',
                    format=
                    '[out]-%(levelname)s:%(message)s'
                    )
    plt.ioff()
    learning_rate_range = 10**np.arange(-4,-1,0.3)
    lambda_1 = 1e-5
    for i in range(5):
        os.makedirs('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, i+1),exist_ok=True)
        print('running 5cv ---------num: %d'%(i+1))
        

        logging.info('input (-1,1) run folds %d'%(i+1))
        # logging.info('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        # logging.info('acc. and distance')
        ci_list=[]
        dataloader=dataset.get_loader(config.path,config.batch_size,'train',dataset_type='cox',kf=i,omics_type=config.omics_type)
        dataloader_val=dataset.get_loader(config.path,config.batch_size,'test',dataset_type='cox',kf=i,omics_type=config.omics_type)

       
        model, loss_nn_all, pvalue_all, c_index_all, c_index_list, acc_train_all, train_max_ci,code_output,os_train,lbl_all,train_p_value,lbl_pre_train,best_max_ci,code_val,os_test,lbl_test_all,best_pvalue,lbl_pred_all_test,OS_test=\
            train(config,dataloader,dataloader_val,config.n_epochs, config.batch_size, config.lr,   True, True, save_state=True,time_str=time_str,idx=i)

        # code_train, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all_train,  OS_train = \
        #         test(model, dataloader_val, 'train',  config.batch_size,  True)
        
        print("[Final] Apply model to training set: c-index: %.10f, p-value: %.10e" % (train_max_ci, train_p_value))
        logging.info("[Final] Apply model to training set: c-index: %.10f, p-value: %.10e" % (train_max_ci, train_p_value))
        # code_test, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all_test,  OS_test = \
        #         test(model, dataloader_val, 'test',  config.batch_size,  True)

        print("[Final] Apply model to testing set: c-index: %.10f, p-value: %.10e" % (best_max_ci, best_pvalue))
        logging.info("[Final] Apply model to testing set: c-index: %.10f, p-value: %.10e" % (best_max_ci, best_pvalue))

        # with open(results_dir_dataset + '/model.pickle', 'wb') as handle:
        #     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # torch.save(model.state_dict(),'result/COX/%s/%d'%(time_str, i+1) + '/model.pth')

        
    
        epochs_list = range(config.n_epochs)
        plt.figure(figsize=(8,4))
        plt.plot(epochs_list, c_index_list['train'], "b--",linewidth=1)
        plt.plot(epochs_list, c_index_list['test'], "g-",linewidth=1)
        plt.legend(['train', 'test'])
        plt.xlabel("epochs")
        plt.ylabel("Concordance index")
        plt.savefig('result/COX_%s/%s_%s/%d'%(config.encoder_type,time_str,config.omics_type, i+1) + "/convergence.png",dpi=300)
        




if __name__=='__main__':
    main()

