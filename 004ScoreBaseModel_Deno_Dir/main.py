import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import numpy as np;
import functools;

class TimeEmbedding(nn.Module):

    "使用高斯随机特征进行 时间的embedding"

    def __init__(self,embed_dim,scale = 30.):
        super().__init__()
        self.w = nn.Parameter(torch.randn(embed_dim // 2)*scale,requires_grad=False)
    def forward(self,x):
        x_proj = x[:,None] * self.w[None,:]*2*np.pi # None 是用来添加一个维度的，x变成了三维
        return torch.cat([torch.sin(x_proj),torch.cos(x_proj)],dim=-1)

class Dense(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.dense(x)[...,None,None] #...就是前面所有维度的意思 ,最后返回的tensor应该是 以第一层为例，[1,32,1,1]

#定义ScoreNet

class ScoreNet(nn.Module):
    def __init__(self,marginal_prob_std,channels=[32,64,128,256],embed_dim = 256):
        super().__init__()
        #时间编码层
        self.embed = nn.Sequential(
            TimeEmbedding(embed_dim),
            nn.Linear(embed_dim,embed_dim)
        )
        #Unet编码层

        self.conv1 = nn.Conv2d(1,channels[0],3,stride=1,bias=False);
        self.dense1 = Dense(embed_dim,channels[0])
        self.gnorm1 = nn.GroupNorm(4,num_channels=channels[0])

        self.conv2  = nn.Conv2d(channels[0],channels[1],3,stride=2,bias=False)
        self.dense2 = Dense(embed_dim,channels[1])
        self.gnorm2 = nn.GroupNorm(32,num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1],channels[2],3,stride=2,bias=False)
        self.dense3 = Dense(embed_dim,channels[2])
        self.gnorm3 = nn.GroupNorm(32,num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2],channels[3],3,stride=2,bias=False)
        self.dense4 = Dense(embed_dim,channels[3])
        self.gnorm4 = nn.GroupNorm(32,num_channels=channels[3])

        #Unet解码器
        self.tconv4 = nn.ConvTranspose2d(channels[3],channels[2],3,stride=2,bias=False)
        self.dense5 = Dense(embed_dim,channels[2])
        self.tgnorm4 = nn.GroupNorm(32,num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2]+channels[2],channels[1],3,stride=2,bias=False,output_padding=1)
        self.dense6 = Dense(embed_dim,channels[1])
        self.tgnorm3 = nn.GroupNorm(32,num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1]+channels[1],channels[0],3,stride=2,bias=False,output_padding=1)
        self.dense7 = Dense(embed_dim,channels[0])
        self.tgnorm2 = nn.GroupNorm(32,num_channels=channels[0])

        #激活函数Swish
        self.act = lambda x:x*torch.sigmoid(x)

        self.tconv1 = nn.ConvTranspose2d(channels[0]+channels[0],1,3,stride=1)
        self.marginal_prob_std = marginal_prob_std

    def forward(self,x,t):
        embed = self.act(self.embed(t))

        h1 = self.conv1(x)
        h1 += self.dense1(embed) #注入时间，后面也是一样的
        h1 =self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

#       解码器部分前向计算
        h = self.tconv4(h4)
        h+= self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        # 按照channel那一个维度拼起来
        h = self.tconv3(torch.cat([h,h3]),dim = 1)
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h =self.tconv2(torch.cat([h,h2],dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat([h,h1],dim=1))

        h = h/self.marginal_prob_std(t)[:,None,None,None] #SDE 里面对每个Unet的结果除以 2范数 平方的期望，希望我们预测分数可以逼近真实的分数


device = 'cuda' #cuda

def marginal_prob_std(t,sigma):
# 定义标准差，这里还把时间t给引入进来了，在分数模型2019中没有时间t，那个t只在郎之万采样的时候用了；这里的t带入就可以得到某一个噪声sigma下，不同t时刻的 扰动数据的 概率分布，也是近似于一个高斯分布；
    t = torch.tensor(t,device = device)
    return torch.sqrt((sigma**(2*t)-1.)/(2.*(np.log(sigma))))

def diffusion_coeff(t,sigma):
    return torch.tensor(sigma**t,device=device)
#噪声
sigma = 25.0
# 构造无参函数：
marginal_prob_std_fn = functools.partial(marginal_prob_std,sigma = sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff,sigma = sigma)

def loss_fn(score_model,x,marginal_prob_std,eps = 1e-5):
    # 生成时间  t  ,生成batchsize 个浮点型 t，shape0就是tensor的number,数量；这个时间t加入的很抽象，他不是对应一个图片，而是对一个batch，然后时间加入进去的时候又是将对应一个batch 的全部的时间  加到单独的每个图片上？？？
    random_t = torch.rand(x.shape[0],device=x.device)*(1 - eps)+eps

    # 基于参数重整化技巧添加扰动数据
    z =torch.randn_like(x)
    #t时刻的标准差
    std = marginal_prob_std(random_t)
    perturbed_x = x + z*std[:,None,None,None]

    # 传入扰动数据和时间t ，得到 分数score。
    score = score_model(perturbed_x,random_t)

# (score*std[:,None,None,None]+z)**2 分数乘以标准差+一个噪声z
    loss = torch.mean(torch.sum((score*std[:,None,None,None]+z)**2,dim=(1,2,3)))

    return loss;


from copy import deepcopy


#对权重 进行指数平滑
class EMA(nn.Module):
    def __init__(self,model,decay = 0.9999,device =None):
        super(EMA,self).__init__();#这是 python2 的写法，3的话不需要写EMA
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)
    def _update(self,model,update_fn):
        with torch.no_grad():
            #zip 把两个[] 组装成字典 {}
            for ema_v,model_v in zip(self.module.state_dict().values(),model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=device)
                ema_v.copy_(update_fn(ema_v,model_v))
    def update(self,model):
        self._update(model,update_fn=lambda e,m:self.decay*e+(1-self.decay)*m)
    def set(self,model):
        self._update(model,update_fn=lambda e,m:m)

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transformers
from torchvision.datasets import MNIST
import tqdm

# 多机多卡一般用DDP-分布式训练
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
device = torch.device("cuda:0")
score_model = score_model.to(device)

n_epechs = 50
batch_size = 32
lr = 1e-4

#根目录下没有数据集就自己下载
dataset = MNIST('.',True,transform=transformers.ToTensor(),download=True);
#开启4个进程去做
data_loader = DataLoader(dataset,batch_size,shuffle=True,num_workers=4)

optimizer = Adam(score_model.parameters(),lr=lr)
tqdm_epochs = tqdm.tqdm(range(n_epechs))

ema = EMA(score_model)
for epoch in tqdm_epochs:
    avg_loss = 0.0;
    num_items = 0
    # x是图片，y是标签，也就是condition，条件，但是这里我们是无条件生成；
    for x,y in data_loader:
        x= x.to(device)
        loss = loss_fn(score_model,x,marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #均值平滑
        ema.update(score_model)

        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    print("Average ScoreMatching Loss:{:5f}".format(avg_loss/num_items))
    #保存模型参数
    torch.save(score_model.state_dict(),f"ckpt_{epoch}.pth")
