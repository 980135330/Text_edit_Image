import torch.nn as nn
from taming.models import vqgan
from omegaconf import OmegaConf
from ..builder  import VAE

# 注册vqgan模型
# VAE._register_module(name = "VQGAN",module_class = vqgan.VQModel)

def exist(val):
    return val is not None

@VAE.register_module()
class VQGAN(nn.Module):
    def __init__(
        self,
        image_size=224,
        compress_ratio=16,
        num_tokens=16384,
        model_cfg=None,
        ckpt_path=None,
    ):
        super(VQGAN, self).__init__()
        self.image_size=image_size

        assert image_size%compress_ratio==0,"image_size is not divisible by compress ratio"
        self.fmap_size=image_size//compress_ratio

        # encoder后总的seq大小
        self.seq_len=self.fmap_size**2
        self.num_tokens=num_tokens


        self.model=None

        # 在这里初始化vqgan模型并读取预训练权重
        if exist(model_cfg):
            model_cfg = OmegaConf.load(model_cfg)
            self.model = vqgan.VQModel(**model_cfg.model.params)
            if exist(ckpt_path):
                self.model.eval().requires_grad_(False)
                self.model.init_from_ckpt(ckpt_path)
    def forward(self,x):
        x = self.model(x)
        return x 
    def encode(self,x):
        x = self.model.encoder(x)
        return x 
    def decode(self,x):
        x = self.model.decoder(x)
        return x 



        
    
    
    
        






    

 


    
        
