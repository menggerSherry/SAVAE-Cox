# Author Xiangyu Meng for Encoder training

import torch
import torch.nn as nn
import sys
import os
sys.path.append("./model")
sys.path.append("./loss")
sys.path.append('./utils')
import dataset
import torch.autograd as autograd
import VGANCox

import argparse
from torch.backends import cudnn
import datetime
import time
import decay_method
import logging
from sklearn.metrics import r2_score
import summary
import distance
import torch.nn.functional as F


cuda = True if torch.cuda.is_available() else False
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class RMSELoss(torch.nn.Module):
    def __init__(self):
        from torch.nn import MSELoss
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

def val_sample(dataloader,gen,idx,total,time_str):
    log = {'rmse': summary.AverageMeter(),'kl':summary.AverageMeter(),'r2':summary.AverageMeter(),'js':summary.AverageMeter()}
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            x=data['exp'].to(device)
            y=data['target'].to(device)
            rec_exp,mu,var=gen(x)
            # target = data['target'].to(device)
            # generate=gen(noise)
            rmse = RMSELoss()(rec_exp,y)

            log['rmse'].update(rmse.item(),config.batch_size)
            kl=F.kl_div(rec_exp.detach().softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
            r2=r2_score(rec_exp.cpu().flatten().detach().numpy(),y.cpu().flatten().detach().numpy())
            js=distance.js_div(rec_exp.detach(),y)
            log['kl'].update(kl,config.batch_size)
            log['r2'].update(r2,config.batch_size)
            log['js'].update(js,config.batch_size)

    logging.info('%d'%idx)
    logging.info('the rmse....%f'%(log['rmse'].avg))
    logging.info('')
    logging.info('the kl distance....%f'%(log['kl'].avg))
    logging.info('')
    logging.info('r2 score: %f'%(log['r2'].avg))
    # logging.info('the gamma....%f'%(gen.encoder.attention.gamma.item()))
    # gen.train()
    return log['r2'].avg,log['kl'].avg,log['js'].avg





def main(config):
    r2_list_val=[]
    kl_list_val=[]
    js_list_val=[]
    os.makedirs('result/GAN_a',exist_ok=True)
    time_str  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
    os.makedirs('result/GAN_a/%s'%time_str,exist_ok=True)

    logging.basicConfig(level=logging.DEBUG,
                    filename='result/GAN_a/%s/%s.log'%(time_str,config.result_log),
                    filemode='a',
                    format=
                    '[out]-%(levelname)s:%(message)s'
                    )

    logging.info('input (-1,1) res  model 1  leaky relu  xvaier')
    # logging.info('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    logging.info('acc. and distance')
    distance=RMSELoss()

    cudnn.benchmark = True
    os.makedirs('saved_models/%s/' % (config.model_path), exist_ok=True)
    dataloader=dataset.get_loader(config.path,config.batch_size,'train')
    dataloader_val=dataset.get_loader(config.path,config.batch_size,'test',shuffle=True)

    gen = VGANCox.SAVAE(config.seq_length,config.sample_length,config.latern_dim,config.dropout)
    gen.train()

   
    # logging.info(gen.encoder.attention.gamma.item())
    prarm=list(gen.named_parameters())
    # print(prarm[1][1].type())
    # print(prarm)
    gen.to(device)
    logging.info('generator arch')
    summary.summary(gen,(config.seq_length,),'gen_1')


    dis=VGANCox.Discriminator(config.seq_length,config.sample_length)
    dis.train()
    dis=dis.to(device)
    logging.info('discriminator arch')
    summary.summary(dis,(config.seq_length,),'dis_1')


    optimizer_G = torch.optim.Adam(gen.parameters(),
                                   lr=config.lr,
                                   betas=(config.beta1, config.beta2))
   
    optimizer_D = torch.optim.Adam(dis.parameters(),
                                             lr=config.lr,
                                             betas=(config.beta1, config.beta2)
                                             )
    

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, 
        lr_lambda=decay_method.LambdaLR(config.n_epochs, config.epoch, config.decay_epoch).step
    )
    
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, 
        lr_lambda=decay_method.LambdaLR(config.n_epochs, config.epoch, config.decay_epoch).step
    )

    if config.epoch != 0:
        gen.load_state_dict(
            torch.load('saved_models/%s/g_%d.pth' % (config.model_path,config.epoch+1)))

        dis.load_state_dict(
            torch.load('saved_models/%s/d_%d.pth' % (config.model_path,config.epoch+1)))
        gen.train()
        
        dis.train()

    prev_time=time.time()
    idx=0
    score=val_sample(dataloader_val,gen,idx=400,total=len(dataloader_val),time_str=time_str)
    for epoch in range(config.epoch,config.n_epochs):
        for i,data in enumerate(dataloader):
            exp=data['exp'].to(device)

            # print(exp.size())
            # print(exp)
            target = data['target'].to(device)

            # exp,target=aug.aug(G_aug,exp,target,config.noise_size,config.batch_size)

            optimizer_D.zero_grad()
             # Generate a batch of images
            fake_imgs,mu,var = gen(exp)

            # Real images
            real_validity = dis(target)
            # print(1,real_validity.size())
            # Fake images
            fake_validity = dis(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(dis, target.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

        # Train the generator every n_critic steps
            if i % config.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs,mu,var = gen(exp)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = dis(fake_imgs)
                g_loss = -torch.mean(fake_validity)
                KLD = -0.5 * (1 + var - mu.pow(2) - var.exp()).sum() / (mu.size(0) * mu.size(1))
                distance_loss=distance(fake_imgs,target)
                g_loss+=20.0*distance_loss
                g_loss+=10.0*KLD
                g_loss.backward()
                optimizer_G.step()


            batch_done = epoch * len(dataloader) + i
            batches_left = config.n_epochs * len(dataloader) - batch_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # 输出
            print(
                '[epoch: %d/%d] [batch: %d/%d] [D loss: %f] [generator loss: %f ; distance loss: %f ] [finished: %f percent] eta: %s'
                % (
                    epoch,
                    config.n_epochs,
                    i,
                    len(dataloader),
                    d_loss.item(),
                    g_loss.item(),
                    distance_loss.item(),
                    (batch_done) / (config.n_epochs * len(dataloader)) * 100,
                    time_left

                )
            )
            
            # if idx%config.sample_interval==0:   
            #     val_sample(dataloader_val,gen,idx)
            idx+=1    
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        if (epoch+1) % config.sample_interval==0:

            score,kl,js=val_sample(dataloader_val,gen,idx,len(dataloader_val),time_str)
            r2_list_val.append(score)
            kl_list_val.append(kl)
            js_list_val.append(js)

        if (epoch+1)>=50 and (epoch+1) % config.checkpoints_interval == 0:
            # torch.save(gen.encoder.state_dict(),'saved_models/%s/g_a_e_%d'%(config.model_path,epoch+1))
            torch.save(gen.state_dict(),'saved_models/%s/g_a_%d'%(config.model_path,epoch+1))
            torch.save(dis.state_dict(),'saved_models/%s/d_a_%d'%(config.model_path,epoch+1))

   

  
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(real_samples.size(0),1).uniform_(0,1)
    alpha = alpha.expand(real_samples.size(0), real_samples.size(1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # print(interpolates.size())
    d_interpolates = D(interpolates)
    # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # print(d_interpolates.size())
    # print(fake.size())
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(
                               d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data_new', help='dataset path')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    parser.add_argument('--n_epochs', type=int, default=200, help='train epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epoch', type=int, default=0, help='epoch')
    parser.add_argument('--decay_epoch', type=int, default=100, help='decay epoch')
    parser.add_argument('--checkpoints_interval', type=int, default=20, help='check')
    parser.add_argument('--lambda_gp', type=int, default=10, help='check')
    parser.add_argument('--sample_interval',type=int,default=2)
    parser.add_argument('--result_log',type=str,default='znorm_res_leakyArelu_xivaer_r2_pixle_dis2048_bn')
    parser.add_argument('--lambda_',type=float,default=10.)
    parser.add_argument('--gp_alpha',type=float,default=10.)
    parser.add_argument('--n_critic',type=int,default=5)
    parser.add_argument('--model_path',type=str,default='SVAEpretrain')
    parser.add_argument('--seq_length',type=int,default=20034)
    parser.add_argument('--noise_size',type=int,default=500)
    parser.add_argument('--hidden_size',type=int,default=1024)
    parser.add_argument('--sample_length',type=int,default=1024)
    parser.add_argument('--dropout',type=bool,default=True)
    parser.add_argument('--loss_type',type=str,default='l1')
    parser.add_argument('--times',type=int,default=1)
    # parser.add_argument('--num_heads',type=int,default=2)
    # parser.add_argument('--code_dim',type=int,default=50)
    parser.add_argument('--label_dim',type=int,default=1)
    parser.add_argument('--latern_dim',type=int,default=256)

    config = parser.parse_args()
    print(config)
    main(config)




