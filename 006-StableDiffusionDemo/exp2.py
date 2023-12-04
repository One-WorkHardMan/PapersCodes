#exp1 里面写了 需要的网络，然后这里呢就写如何做训练：

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.utils.data as tud
import torch.nn.functional as func


class FakeDataset(tud.Dataset):

    def __init__(self,src_shape = (17,128,128),dst_shape = (1,128,128)):
        self.src_shape = src_shape;
        self.dst_shape = dst_shape;
        self.sample_count = 10000;

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index:int):
        # 生成 随机数作为 假数据
        xx = torch.rand(self.src_shape,dtype=torch.float32)
        yy = torch.rand(self.dst_shape,dtype=torch.float32)
        return xx,yy
#-------------------------------------------------------------------------------------------------------------------------
# vae的训练过程
import ex1
def train_vea():
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    default_dtype = torch.float32

    # 模型初始化
    # init model
    vae  = ex1.VAE(17,latent_dim=128)
    # init weight
    vae.to(default_device,dtype=default_dtype)


    # 设置优化器
    optimizer = AdamW(vae.parameters(),lr=1e-3)

    ds =FakeDataset()
    dl = tud.DataLoader(ds,32,shuffle=True,drop_last=True,num_workers=0)

    # enumerate(dl) 会返回 Batch的id，以及这个batch 的 xx数据的矩阵和标签yy的矩阵
    for batch,(xx,yy) in enumerate(dl):
        xx = xx.to(default_device,dtype=default_dtype)

        #step
        optimizer.zero_grad()
        z,kl_loss = vae.encode(xx,return_loss=True)
        x_hat = vae.decode(z)

        # 这个就是重构 loss ， 但是这里和我笔记里面记录的不太一样因为，笔记里面的Decoder输出的其实还是均值和方差，是要是均值和初始x的距离最接近，这里的模型写的是Decoder输出的就是一个向量，也就是恢复之后的值。
        mse_loss = func.mse_loss(x_hat,xx)

        loss = kl_loss+mse_loss;

        loss.backword()

        optimizer.step()
#-------------------------------------------------------------------------------------------------------------------------
# vqvae 的 训练过程

def train_vq_vae():
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    default_dtype = torch.float32

    # 模型初始化
    # init model
    vae  = ex1.VQVAE(17,latent_dim=128)
    # init weight 权重初始化这里没写而已
    vae.to(default_device,dtype=default_dtype)


    # 设置优化器
    optimizer = AdamW(vae.parameters(),lr=1e-3)

    ds =FakeDataset()
    dl = tud.DataLoader(ds,32,shuffle=True,drop_last=True,num_workers=0)

    # enumerate(dl) 会返回 Batch的id，以及这个batch 的 xx数据的矩阵和标签yy的矩阵
    for batch,(xx,yy) in enumerate(dl):
        xx = xx.to(default_device,dtype=default_dtype)

        #step
        optimizer.zero_grad()
        z,vq_loss = vae.encode(xx,return_loss=True)
        x_hat = vae.decode(z)

        # 这个就是重构 loss ， 但是这里和我笔记里面记录的不太一样因为，笔记里面的Decoder输出的其实还是均值和方差，是要是均值和初始x的距离最接近，这里的模型写的是Decoder输出的就是一个向量，也就是恢复之后的值。
        mse_loss = func.mse_loss(x_hat,xx)

        # vqvae的损失函数就是 一个重构loss 和 codebook的loss
        loss = vq_loss+mse_loss;

        loss.backword()

        optimizer.step()

#-------------------------------------------------------------------------------------------------------------------------
#VQ_GAN




#-------------------------------------------------------------------------------------------------------------------------





















