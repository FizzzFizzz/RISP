import torch.nn as nn
import torch
import hdf5storage
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import os
from natsort import os_sorted
import sys 
# sys.path.append("..")
sys.path.append("utils/")
import utils_image as util
import utils_logger
import utils_deblur as deblur
import utils_sr
from argparse import ArgumentParser
from denoiser_model import *



def data_fidelity_init(self, kernel = None, mask = None, init = None):
    if self.Pb == 'deblurring':
        # Initialization of Gradient operator
        fft_k = deblur.p2o(kernel, init.shape[-2:])
        fft_kH = torch.conj(fft_k)
        abs_k = fft_kH * fft_k
        self.abs_k = abs_k
        self.fft_kH = fft_kH
        # Initialization of Proximal operator
        self.k_tensor = torch.tensor(kernel).to(self.device)
        self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(init, self.k_tensor, self.sf)
    elif self.Pb == 'SR':
        # Initialization of Proximal operator
        self.k_tensor = torch.tensor(kernel).to(self.device)
        self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(init, self.k_tensor, self.sf)
    elif self.Pb == 'inpainting':
        self.M = array2tensor(mask).to(self.device)
    elif self.Pb == 'ODT':
        # print('ODT loves you~')
        from utils.inverse_scatter import InverseScatter
        self.scatter_op = InverseScatter(
            Lx=0.18, Ly=0.18, Nx=128, Ny=128, 
            wave=6, numRec=180, numTrans=20,
            sensorRadius=1.6, svd=False
        )
    else:
        raise ValueError("Forward Model not implemented")


def compute_data_grad(self, x, obs):
    if self.noise_model == 'gaussian':
        if self.Pb == 'deblurring':
            data_grad = self.abs_k * deblur.fftn(x) - self.fft_kH * deblur.fftn(obs)
            data_grad = torch.real(deblur.ifftn(data_grad))
        elif self.hparams.degradation_mode == 'inpainting':
            data_grad = 2 * self.M * (x - obs)
        elif self.Pb == 'ODT':
            with torch.enable_grad():
                v = x.clone() 
                v.requires_grad_()
                t = v
                t = t.type(torch.cuda.FloatTensor)
                uscat_pred = self.scatter_op.forward(t, unnormalize=False)
                difference = uscat_pred - obs
                norm = torch.sum(torch.abs(difference)**2)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=v)[0]
                data_grad = norm_grad
        else:
            raise ValueError("Forward Model not implemented")
    else :  
        ValueError('Forward Model noise model not implemented')
    return data_grad

def data_fidelity_prox_step(self, x, y, stepsize):
    '''
    Calculation of the proximal step on the data-fidelity term f
    '''
    if self.noise_model == 'gaussian':
        if self.Pb == 'deblurring' or self.Pb == 'SR':
            px = utils_sr.prox_solution_L2(x, self.FB, self.FBC, self.F2B, self.FBFy, stepsize, self.sf)
        elif self.Pb == 'inpainting':
            if self.hparams.noise_level_img > 1e-2:
                px = (stepsize*self.M*y + x)/(stepsize*self.M+1)
            else :
                px = self.M*y + (1-self.M)*x
        elif self.Pb == 'ODT':
            input = x
            uu = input.clone()
            vv = input.clone()
            with torch.enable_grad():
                for _ in range(5):
                    uu = vv.clone()
                    uu = uu.detach().requires_grad_(True)
                    uscat_pred = self.scatter_op.forward(uu, unnormalize=False)
                    difference = uscat_pred - y
                    norm = 0.5*(self.lamb)*torch.sum(torch.abs(difference)**2) + 0.5*torch.sum(torch.abs(uu-input)**2)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=uu)[0]
                    data_grad = norm_grad
                    vv -= data_grad*0.1
                    vv = torch.clamp(vv,-0,1)
            x = uu
            # by wdl
            x = torch.clamp(x,-0,1)
            px = x
        else:
            ValueError('Forward Model degradation not implemented')
    else :  
        ValueError('Forward Model noise model not implemented')
    return px

