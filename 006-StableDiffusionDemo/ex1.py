








import torch;
import torch.nn as nn;
#---------------------------------------------------------------------------------------------------------------------------
#现在都是些分块来进行，训练不是直接写一个网络
#常规的卷积块
class ConvBlock(nn.Module):
    def __init__(self,num_channels:int):
        super.__init__();
        self.layers = nn.Sequential(
            nn.Conv2d (num_channels,num_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        h = self.layers(inputs)
        return h
#---------------------------------------------------------------------------------------------------------------------------
#残差连接
class ResBlock(nn.Module):
    def __init__(self,num_channels:int):
        super(ResBlock, self).__init__();
        self.residual = nn.Sequential(
            # nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_channels),
            # nn.ReLU(),
            #
            # nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_channels),
            # nn.ReLU(),

            #不过后来何开明做了个改进:效果更好
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        h = inputs + self.residual(inputs)
        return h
#---------------------------------------------------------------------------------------------------------------------------
#下采样和上采样 简化版Unet ，没有 前后相加，但是有残差；
class AutoEncoder(nn.Module):
    def __init__(self,num_channels = 3,ch:int = 64):
        super(AutoEncoder, self).__init__()

        """
            模型的输入，输出，需要单独拿出来写，这样可以增加模型复用性，比如我这里Encoder和Decoder训练好了
            我可以只改变 convin 的通道，和输出通道，中间的参数不动
            做FineTune。
        """

        self.conv_in = nn.Conv2d(num_channels,1*ch,kernel_size=3,stride=1,padding=1)

        self.Encoder = nn.Sequential(
            #下采样Stage 1
            nn.Sequential(
                ResBlock(1*ch),
                ResBlock(1*ch)
            ),
            # 220 的好处就是，刚好长宽缩小一般，通道增大一倍
            #Stage2
            nn.Conv2d(1*ch,2*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),
            # Stage3
            nn.Conv2d(2*ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4 * ch),
                ResBlock(4 * ch)
            ),
            # Stage4
            nn.Conv2d(4 * ch, 8 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(8* ch),
                ResBlock(8* ch)
            ),
        )

        #上采样
        self.Decoder = nn.Sequential(
            #Stage4
            nn.Sequential(
                ResBlock(8 * ch),
                ResBlock(8 * ch)
            ),

            nn.ConvTranspose2d(8 * ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4* ch),
                ResBlock(4 * ch)
            ),

            nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),

            nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(1 * ch),
                ResBlock(1 * ch)
            )
        )

        self.conv_out =  nn.Conv2d(1*ch,num_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        h = self.conv_in(inputs)
        h = self.Encoder(h)
        h = self.Decoder(h)
        h = self.conv_out(h)
        return h
#---------------------------------------------------------------------------------------------------------------------------

"""
    VAE 来试试 Diffusers这个包 导入预训练好的模型吧：
"""

from diffusers.models.vae import DiagonalGaussianDistribution

class VAE(nn.Module):
    def __init__(self,num_channels = 3,latent_dim:int=128 ,ch:int = 64):
        super(VAE, self).__init__()

        """
            模型的输入，输出，需要单独拿出来写，这样可以增加模型复用性，比如我这里Encoder和Decoder训练好了
            我可以只改变 convin 的通道，和输出通道，中间的参数不动
            做FineTune。
        """

        self.conv_in = nn.Conv2d(num_channels,1*ch,kernel_size=3,stride=1,padding=1)

        self.Encoder = nn.Sequential(
            #下采样Stage 1
            nn.Sequential(
                ResBlock(1*ch),
                ResBlock(1*ch)
            ),
            # 220 的好处就是，刚好长宽缩小一般，通道增大一倍
            #Stage2
            nn.Conv2d(1*ch,2*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),
            # Stage3
            nn.Conv2d(2*ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4 * ch),
                ResBlock(4 * ch)
            ),
            # Stage4
            nn.Conv2d(4 * ch, 8 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(8* ch),
                ResBlock(8* ch)
            ),
        )

        #隐含变量z
        self.proj1 = nn.Conv2d(8 * ch,2*latent_dim,kernel_size=1,stride=1,padding=0),
        #z
        #为啥是 latent_dim， VAE里面，经过Encoder之后，然后参数重整化之后向量的维度就是 上面的 2*latent_dim 一半。
        self.proj2 = nn.Conv2d(latent_dim,8*ch,kernel_size=1,stride=1,padding=0),



        #上采样
        self.Decoder = nn.Sequential(
            #Stage4
            nn.Sequential(
                ResBlock(8 * ch),
                ResBlock(8 * ch)
            ),

            nn.ConvTranspose2d(8 * ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4* ch),
                ResBlock(4 * ch)
            ),

            nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),

            nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(1 * ch),
                ResBlock(1 * ch)
            )
        )

        self.conv_out =  nn.Conv2d(1*ch,num_channels,kernel_size=3,stride=1,padding=1)

    def encode(self,inputs:torch.Tensor,sampling:bool = False,return_loss:bool = False):
        h = self.conv_in(inputs)
        h = self.Encoder(h)
        h = self.proj1(h)

        # 隐变量传进去分成两部分：
        latent_dist = DiagonalGaussianDistribution(h)

        if sampling:
        # 参数重整化的方式进行采样
            return latent_dist.sample()

        # 训练阶段
        elif return_loss:
            #VAE 的 KL—loss
            kl_loss = latent_dist.kl()
            return latent_dist.sample(),kl_loss
        else:
            return latent_dist.mode()

    def decode(self,inputs)->torch.Tensor:
        h = self.proj2(inputs)
        h = self.Decoder(h)
        return h


    # def forward(self,inputs:torch.Tensor)->torch.Tensor:
    #     h = self.conv_in(inputs)
    #     h = self.Encoder(h)
    #     h = self.Decoder(h)
    #     h = self.conv_out(h)
    #     return h

#------------------------------------------------------------------------------------------------------------------------

from diffusers.models.vae import VectorQuantizer
# 引入向量量化器
class VQVAE(nn.Module):
    def __init__(self,num_channels = 3,latent_dim:int=128 ,ch:int = 64):
        super(VQVAE, self).__init__()

        """
            模型的输入，输出，需要单独拿出来写，这样可以增加模型复用性，比如我这里Encoder和Decoder训练好了
            我可以只改变 convin 的通道，和输出通道，中间的参数不动
            做FineTune。
        """

        self.conv_in = nn.Conv2d(num_channels,1*ch,kernel_size=3,stride=1,padding=1)

        self.Encoder = nn.Sequential(
            #下采样Stage 1
            nn.Sequential(
                ResBlock(1*ch),
                ResBlock(1*ch)
            ),
            # 220 的好处就是，刚好长宽缩小一般，通道增大一倍
            #Stage2
            nn.Conv2d(1*ch,2*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),
            # Stage3
            nn.Conv2d(2*ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4 * ch),
                ResBlock(4 * ch)
            ),
            # Stage4
            nn.Conv2d(4 * ch, 8 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(8* ch),
                ResBlock(8* ch)
            ),
        )

        #隐含变量z
        self.proj1 = nn.Conv2d(8 * ch,latent_dim,kernel_size=1,stride=1,padding=0),
        #z
        self.vq = VectorQuantizer(n_e=8192,vq_embed_dim=latent_dim,beta=0.2,legacy=False)

        self.proj2 = nn.Conv2d(latent_dim,8*ch,kernel_size=1,stride=1,padding=0),



        #上采样
        self.Decoder = nn.Sequential(
            #Stage4
            nn.Sequential(
                ResBlock(8 * ch),
                ResBlock(8 * ch)
            ),

            nn.ConvTranspose2d(8 * ch,4*ch,kernel_size=2,stride=2,padding=0),
            nn.Sequential(
                ResBlock(4* ch),
                ResBlock(4 * ch)
            ),

            nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),

            nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ResBlock(1 * ch),
                ResBlock(1 * ch)
            )
        )

        self.conv_out =  nn.Conv2d(1*ch,num_channels,kernel_size=3,stride=1,padding=1)

    def encode(self,inputs:torch.Tensor,sampling:bool = False,return_loss:bool = False):
        h = self.conv_in(inputs)
        h = self.Encoder(h)
        h = self.proj1(h)

        #return z_q, loss, (perplexity, min_encodings, min_encoding_indices) , 这个VQ 要返回几个东西，ze量化之后的结果，loss
        #min_encoding_indices 码本的索引；
        z_q,loss,_ = self.vq(h)
        if sampling:
            return z_q
        elif return_loss:
            #train 训练的时候一定要用到loss
            return z_q,loss
        else:
            return h


    def decode(self,inputs)->torch.Tensor:
        h = self.proj2(inputs)
        h = self.Decoder(h)
        return h

#-------------------------------------------------------------------------------------------------------------------------
#VA-GAN 的 patch GAN部分
# class Patch_GAN_Discriminator(nn.Module):
#     def __init__(self,in_channels:int,num_channels:int = 64):
#         super.__init__()
#         self.layers = nn.Sequential(
#             nn.Sequential(
#                 nn.Conv2d()
#
#             ),
#
#
#         )


#-------------------------------------------------------------------------------------------------------------------------
#U-Net 模型 : 这是Unet+残差块

class UNet(nn.Module):
    def __init__(self,num_channels:int=3,ch:int = 64):
        super(UNet, self).__init__()
        self.conv_in = nn.Conv2d(num_channels,1*ch,kernel_size=3,stride=1,padding=1)

        self.Encoder = nn.ModuleList([
            # 下采样Stage 1
            nn.Sequential(
                ResBlock(1 * ch),
                ResBlock(1 * ch)
            ),
            # 220 的好处就是，刚好长宽缩小一般，通道增大一倍
            # Stage2

            nn.Sequential(
                nn.Conv2d(1 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
                ResBlock(2 * ch),
                ResBlock(2 * ch)
            ),
            # Stage3

            nn.Sequential(
                nn.Conv2d(2 * ch, 4 * ch, kernel_size=2, stride=2, padding=0),
                ResBlock(4 * ch),
                ResBlock(4 * ch)
            ),
            # Stage4

            nn.Sequential(
                nn.Conv2d(4 * ch, 8 * ch, kernel_size=2, stride=2, padding=0),
                ResBlock(8 * ch),
                ResBlock(8 * ch)
            ),

        ])

        #unet的中间层
        self.middle = nn.Sequential(
            nn.Conv2d(8*ch,32*ch,kernel_size=2,stride=2,padding=0),

            ResBlock(32*ch),
            ResBlock(32*ch),

            nn.ConvTranspose2d(32*ch,8*ch,kernel_size=2,stride=2,padding=0),


        ) ,

        self.Decoder = nn.ModuleList([
            # Stage4

            nn.Sequential(
                # 这里包括上面AE部分的上采样，这几个参数都非常抽象，220 还原成两倍的Input尺寸；
                nn.Conv2d(16*ch,8*ch,kernel_size=3,stride=1,padding=1),
                ResBlock(8 * ch),
                ResBlock(8 * ch),
                nn.ConvTranspose2d(8 * ch, 4 * ch, kernel_size=2, stride=2, padding=0),
            ),


            nn.Sequential(
                nn.Conv2d(8 * ch, 4* ch, kernel_size=3, stride=1, padding=1),
                ResBlock(4 * ch),
                ResBlock(4 * ch),
                nn.ConvTranspose2d(4* ch, 2 * ch, kernel_size=2, stride=2, padding=0),
            ),


            nn.Sequential(
                # nn.ConvTranspose2d(4 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
                nn.Conv2d(8 * ch, 4 * ch, kernel_size=3, stride=1, padding=1),
                ResBlock(2 * ch),
                ResBlock(2 * ch),
                nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2, padding=0),
            ),


            nn.Sequential(
                nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2, padding=0),
                ResBlock(1 * ch),
                ResBlock(1 * ch)
            )


        ]),

        self.conv_out = nn.Conv2d(1*ch,num_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        shotcuts = []
        h = self.conv_in(inputs)
        for m in self.Encoder:
            h = m(h)
            #将每一层的输出保存下来，倒序保存下来
            shotcuts.insert(0,h)

        h = self.middle(h)

        for m,r in zip(self.Decoder,shotcuts):
            h = torch.cat([r,h],dim = 1)
            h = m(h)
        h = self.conv_out(h)
        return h

#-------------------------------------------------------------------------------------------------------------------------
# 在StableDiffusion中，我们的Unet加入了CrossAttention机制，将条件Condition通过CrossAttention加入到了Unet进行训练
# 这里呢就是如果继承了TwoWaysModule，那么就是交叉注意力，否则就是自注意力；
class TwoWaysModule(object):
    pass

class TwoWaysSequential(nn.Module):
    def __init__(self,*modules):
        super().__init__()

        # 只有通过ModuleList 也就是继承了 nn.Mudule 的才能保存权重，如果直接 self.module_list = mudules 是不行的；
        self.module_list = nn.ModuleList(modules)

    # 交叉注意力 就是两个序列之间的关系：
    def forward(self,inputs:torch.Tensor,conditions:torch.Tensor)->torch.Tensor:
        h = inputs
        for m in self.module_list:
            if isinstance(m,TwoWaysModule):
                h = m(h,conditions)
            else:
                h = m(h)
        return h


# 交叉注意力块
class CrossAttentionBlock(nn.Module,TwoWaysModule):
    def __init__(self,num_channels:int ,
                 condition_channels:int,
                 num_heads:int = 8,
                 layer_scale_init:float = 1e-6 #这个值一般利于模型的收敛
                 ):
        super(CrossAttentionBlock, self).__init__()
        #可学习的对角矩阵，控制 输入和输出 进行残差的比例
        self.layer_scale = nn.Parameter(torch.full([num_channels,1,1],layer_scale_init))

        self.layer_norm = nn.GroupNorm(1,num_channels) # 分组为1 就是 ln

        #定义多头注意力：
        self.attention = nn.MultiheadAttention(
            embed_dim=num_channels,
            kdim=condition_channels,
            vdim=condition_channels,
            num_heads=num_heads,
            #QKV是三维Tensor，Batchfirst 就是 （N,L,C）的形式；
            batch_first=True
        )

    def forward(self,inputs:torch.Tensor,conditions:torch.Tensor)->torch.Tensor:
        #inputs [N,C,H,W]
        #conditions [N,L,C]

        h = self.layer_norm(inputs)
        _n,_c,_h,_w = h.shape

        h = h.reshape(_n,_c,_h*_w) #这里 L =_h*_w，说明整个序列的长度是图像的分辨率，然后序列的每个向量的维度是图像的channels个数；
        h =torch.swapdims(h,1,2) # [N,L,C]

        #只用传入QKV就行
        h,_ = self.attention(h,conditions,conditions)
        h = torch.swapdims(h, 1, 2)  # [N,C,L]

        h = h.reshape(_n, _c, _h , _w) #还原成[N,C,H,W]

        h = inputs + self.layer_scale * h
        return h



#------------------------------------------------------------------------------------------------------------------------
#残差+crossattention+Unet


class Attention_UNet_with_Condition(nn.Module):
    def __init__(self,num_channels:int=3,condition_channels:int =128,ch:int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(num_channels,1*ch,kernel_size=3,stride=1,padding=1)

        self.Encoder = nn.ModuleList([
            # 下采样Stage 1
            TwoWaysSequential(
                ResBlock(1 * ch),
                CrossAttentionBlock(1*ch,condition_channels),
                ResBlock(1 * ch),
                CrossAttentionBlock(1 * ch, condition_channels),
            ),
            # 220 的好处就是，刚好长宽缩小一般，通道增大一倍
            # Stage2

            TwoWaysSequential(
                nn.Conv2d(1 * ch, 2 * ch, kernel_size=2, stride=2, padding=0),
                ResBlock(2 * ch),
                CrossAttentionBlock(2 * ch, condition_channels),
                ResBlock(2 * ch),
                CrossAttentionBlock(2 * ch, condition_channels),
            ),
            # Stage3

            TwoWaysSequential(
                nn.Conv2d(2 * ch, 4 * ch, kernel_size=2, stride=2, padding=0),
                ResBlock(4 * ch),
                CrossAttentionBlock(4 * ch, condition_channels),
                ResBlock(4 * ch),
                CrossAttentionBlock(4 * ch, condition_channels),
            ),
            # Stage4 在StableDiffusion里面，这个第四阶段是没有加注意力的；

            TwoWaysSequential(
                nn.Conv2d(4 * ch, 8 * ch, kernel_size=2, stride=2, padding=0),
                ResBlock(8 * ch),
                # CrossAttentionBlock(8 * ch, condition_channels),
                ResBlock(8 * ch),
                # CrossAttentionBlock(8 * ch, condition_channels),
            ),

        ])

        #unet的中间层
        self.middle = nn.ModuleList([

            TwoWaysSequential(

                nn.Conv2d(8 * ch, 32 * ch, kernel_size=2, stride=2, padding=0),
                ResBlock(32 * ch),
                CrossAttentionBlock(32 * ch, condition_channels),
                ResBlock(32 * ch),
                CrossAttentionBlock(32 * ch, condition_channels),
                nn.ConvTranspose2d(32 * ch, 8 * ch, kernel_size=2, stride=2),
            )
        ])

        self.Decoder = nn.ModuleList([
            # Stage4,和上面一样，第四阶段没有计算注意力。

            TwoWaysSequential(
                # 这里包括上面AE部分的上采样，这几个参数都非常抽象，220 还原成两倍的Input尺寸;
                nn.Conv2d(16 * ch, 8 * ch, kernel_size=3, stride=1, padding=1),
                ResBlock(8 * ch),
                ResBlock(8 * ch),
                nn.ConvTranspose2d(8 * ch, 4 * ch, kernel_size=2, stride=2),

            ),


            TwoWaysSequential(
                nn.Conv2d(8 * ch, 4 * ch, kernel_size=3, stride=1, padding=1),
                ResBlock(4 * ch),
                CrossAttentionBlock(4 * ch, condition_channels),
                ResBlock(4 * ch),
                CrossAttentionBlock(4 * ch, condition_channels),
                nn.ConvTranspose2d(4* ch, 2 * ch, kernel_size=2, stride=2),
            ),


            TwoWaysSequential(
                nn.Conv2d(4 * ch, 2 * ch, kernel_size=3, stride=1, padding=1),
                ResBlock(2 * ch),
                CrossAttentionBlock(2 * ch, condition_channels),
                ResBlock(2 * ch),
                CrossAttentionBlock(2 * ch, condition_channels),
                nn.ConvTranspose2d(2 * ch, 1 * ch, kernel_size=2, stride=2),
            ),


            TwoWaysSequential(
                nn.Conv2d(2 * ch, 1 * ch, kernel_size=3, stride=1, padding=1),
                ResBlock(1 * ch),
                CrossAttentionBlock(1 * ch, condition_channels),
                ResBlock(1 * ch),
                CrossAttentionBlock(1 * ch, condition_channels),

            )


        ])

        self.conv_out = nn.Conv2d(1*ch,num_channels,kernel_size=3,stride=1,padding=1)

    def forward(self,inputs:torch.Tensor,conditions:torch.Tensor)->torch.Tensor:
        shotcuts = []
        h = self.conv_in(inputs)
        for m in self.Encoder:
            h = m(h,conditions)
            #将每一层的输出保存下来，倒序保存下来
            shotcuts.insert(0,h)

        for m in self.middle:
            h = m(h,conditions)

        for m,r in zip(self.Decoder,shotcuts):

            h = torch.cat([r,h],dim = 1)
            h = m(h, conditions)

        h = self.conv_out(h)
        return h

#------------------------------------------------------------------------------------------------------------------------

def _test():
    xx = torch.rand([4,17,128,128])
    conditions = torch.rand([4,1,128])
    net = Attention_UNet_with_Condition(num_channels=17,condition_channels=128)
    yy = net(xx,conditions)
    print(yy.shape)


if __name__ == '__main__':
    _test()














































