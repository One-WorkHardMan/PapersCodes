
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.utils.data as tud
import torch.nn.functional as tnf
from diffusers.schedulers import DDPMScheduler
from diffusers.models.embeddings import get_timestep_embedding
import ex1

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

# 在DDPM或者IDDPM中我们加入条件没有使用CrossAttention，我们时间步数的生成都是直接加入数据里面，但是这里要计算交叉注意力分数所以得转换成才能NLC的形式；
# 这里传的不是离散的zq而是h，一个tensor.
def make_conditions(timesteps:torch.Tensor,ElseCondition:torch.Tensor = None,embedding_dim = 128)->torch.Tensor:
    assert timesteps.ndim == 1

    timesteps_enbedding = get_timestep_embedding(timesteps,embedding_dim,max_period=10000) # [N C]
    timesteps_enbedding = timesteps_enbedding[:,None,:] #[N1C]

    if ElseCondition is not None:
        assert ElseCondition.shape[1] == embedding_dim;
        ElseCondition_embed = torch.flatten(ElseCondition,2) #[N,C,H*W]
        ElseCondition_embed = torch.swapdims(ElseCondition,1,2)
        condition_embedding = torch.cat([timesteps_enbedding,ElseCondition_embed],dim=1) #[B,1+L,C]
    else:
        condition_embedding = timesteps_enbedding

    #给序列添加 位置编码，然后加入条件里面去；
    length = condition_embedding.shape[1]
    positions = torch.arange(0,length,device=condition_embedding.device,dtype = condition_embedding.dtype)
    positions_embedding = get_timestep_embedding(positions,embedding_dim,max_period=10000) # [1+L,C]
    positions_embedding = positions_embedding[None,...]

    condition_embedding  = condition_embedding + positions_embedding

    return condition_embedding;









#Train_Diffusion
def train_diffusion():
    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32

    #init model
    condition_unet = ex1.Attention_UNet_with_Condition(128,condition_channels=512)
    condition_unet.to(default_device,default_type)
    optimozer = AdamW(condition_unet.parameters(),lr=1e-3,weight_decay=0.05)


    # 这里假装编码器已经训练好了。Load Encoder.
    #条件编码器 , ldm 的条件编码和图像编码应该都是用VQ-VAE，Dalle2就是用的CLIP，显然CLIP更加聪明。
    con_ae = ex1.VQVAE(num_channels=128,latent_dim=512)
    con_ae.to(default_device,default_type)
    #加载预训练好的模型 load form checkpoints
    con_ae.requires_grad_(False)
    con_ae.eval()

    #图像编码器，编码到隐空间
    img_ae = ex1.VQVAE(num_channels=17,latent_dim=128)
    img_ae.to(default_device,default_type)
    img_ae.requires_grad_(False)
    img_ae.eval()

    #Init Dataset
    ds = FakeDataset()
    dataloader = tud.DataLoader(ds,batch_size=32,shuffle=True,drop_last=True,num_workers=8)

    #Init DDOM-Model
    DDPM_Scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=False,

    )
    for batch,(xx,yy) in enumerate(dataloader):
        xx = xx.to(default_device,default_type)
        yy = yy.to(default_device,default_type)

        optimozer.zero_grad()
        with torch.no_grad():
            src_latent = con_ae.encode(xx)
            tgt_latent = img_ae.encode(yy)

            #make condition , (tgt_latent.shape[0],)就是每个图像一个时间t。
            timesteps = torch.randint(0,DDPM_Scheduler.num_train_timesteps,(tgt_latent.shape[0],),device=default_device,dtype=torch.long)
            # glide 里面的做法，随机训练有条件和无条件两种情况而不是进行分开训练，对有条件还要单独训练一次。
            if batch % 2 == 0 :
                conditions = make_conditions(timesteps,src_latent,embedding_dim = 512)
            else:
                conditions = make_conditions(timesteps,None,embedding_dim = 512)

            nosie = torch.randn_like(tgt_latent)

            # 这里的噪声和时间t 得对应哦；
            tgt_latent = DDPM_Scheduler.add_noise(tgt_latent,nosie,timesteps)

        outputs = condition_unet(tgt_latent,conditions)
        loss = tnf.mse_loss(outputs,nosie)
        loss.backward()
        optimozer.step()
        loss = loss.detach().cpu().numpy()
        print("Bacth{:10d} | Loss:{:8.4f}".format(batch,loss))

#采样阶段

def sampling_diffusion():
    # 看示例，一模一样。
    return None



def main():
    # train_vae()
    # train_diffusion()
    sampling_diffusion()

if __name__ == '__main__':
    main()
















