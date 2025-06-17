import torch
import sys 
sys.path.append("utils/")
import utils_image as util
import utils_logger
import utils_deblur as deblur
sys.path.append("utils/models/") 
from models.network_unet import UNetRes as Net

class Drunet_running(torch.nn.Module):# DRUNet model definition 
    def __init__(self, model_path, n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose", bias=False):
        super(Drunet_running, self).__init__()
        self.model = Net(in_nc=n_channels+1, out_nc=n_channels, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode, bias=bias)
        self.model.load_state_dict(torch.load(model_path), strict=True)
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)    

    def forward(self, x, sigma, device):
        '''
        x : image with values in [0, 1]
        sigma : standard deviation of denoising in [0, 1]
        '''
        sigma = float(sigma)
        sigma_div_255 = torch.FloatTensor([sigma]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)
        x = torch.cat((x, sigma_div_255), dim=1)
        return self.model(x)