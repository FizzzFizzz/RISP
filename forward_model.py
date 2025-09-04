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

def gen_data(self, clean_image, sigma = None, seed=0):
    """
    Generate the degradate observation
    """
    # For reproducibility
    gen = torch.Generator()
    gen.manual_seed(seed)
    if self.Pb == "deblurring":
        fft_k = deblur.p2o(self.kernel, clean_image.shape[-2:])
        temp = fft_k * deblur.fftn(clean_image)
        observation_without_noise = torch.abs(deblur.ifftn(temp))
        noise = torch.normal(torch.zeros(observation_without_noise.size()), torch.ones(observation_without_noise.size()), generator = gen)*sigma / 255
        return (observation_without_noise + noise).to(self.device)
    elif self.Pb == "inpainting":
        probs = torch.full((clean_image.shape[2],clean_image.shape[3]), self.p)
        mask = torch.bernoulli(probs, generator=gen)
        mask = mask.unsqueeze(0).unsqueeze(0).to(self.device)
        observation_without_noise = clean_image*mask
        noise = (torch.normal(torch.zeros(observation_without_noise.size()), torch.ones(observation_without_noise.size()), generator = gen)*sigma / 255).to(self.device)
        observation = observation_without_noise + noise
        return observation, mask
    elif self.Pb == "MRI":
        M = genMask(clean_image.shape[-2:], self.numLines, device=self.device)
        observation_without_noise = M[None,None,:,:] * fft2c(clean_image)
        noise = (torch.normal(torch.zeros(observation_without_noise.size()), torch.ones(observation_without_noise.size()), generator = gen)*sigma / 255).to(self.device)
        observation = observation_without_noise + noise
        pseudo_inverse = torch.real(ifft2c(mask*observation.clone()))
        return observation, M, pseudo_inverse
    elif self.Pb == "speckle":
        observation = injectspeckle_amplitude_log(clean_image, L = self.L, device = self.device, gen = gen)
        return observation
    else:
        raise ValueError("Forward Model not implemented")


def data_fidelity_init(self, init = None):
    if self.Pb == 'deblurring':
        # Initialization of Gradient operator
        fft_k = deblur.p2o(self.kernel, init.shape[-2:])
        fft_kH = torch.conj(fft_k)
        abs_k = fft_kH * fft_k
        self.abs_k = abs_k
        self.fft_kH = fft_kH
        # Initialization of Proximal operator
        self.k_tensor = torch.tensor(self.kernel).to(self.device)
        self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(init, self.k_tensor, self.sf)
    elif self.Pb == 'SR':
        # Initialization of Proximal operator
        self.k_tensor = torch.tensor(self.kernel).to(self.device)
        self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(init, self.k_tensor, self.sf)
    elif self.Pb == 'inpainting':
        self.M = (self.mask).clone()
    elif self.Pb == "MRI":
        self.neg_M = torch.ones(self.M.shape).to(self.device) - 1*self.M
    elif self.Pb == 'ODT':
        from utils.inverse_scatter import InverseScatter
        self.scatter_op = InverseScatter(
            Lx=0.18, Ly=0.18, Nx=128, Ny=128, 
            wave=6, numRec=180, numTrans=20,
            sensorRadius=1.6, svd=False
        )
    elif self.Pb == "speckle":
        self.f_speckle = lambda u, y : torch.sum(self.L * (u + torch.exp(y - u)))
    else:
        raise ValueError("Forward Model not implemented")


def compute_data_grad(self, x, obs):
    if self.noise_model == 'gaussian':
        if self.Pb == 'deblurring':
            data_grad = self.abs_k * deblur.fftn(x) - self.fft_kH * deblur.fftn(obs)
            data_grad = torch.real(deblur.ifftn(data_grad))
        elif self.Pb == 'inpainting':
            data_grad = 2 * self.M * (x - obs)
        elif self.Pb == 'MRI':
            data_grad = torch.real(ifft2c(self.M * (fft2c(x) - obs)))
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
    elif self.noise_model == 'speckle':
        return self.L*(1-torch.exp(obs-x))
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
            if self.sigma_obs > 1e-2:
                px = (stepsize*self.M*y + x)/(stepsize*self.M+1)
            else :
                px = self.M*y + (1-self.M)*x
        elif self.Pb == 'MRI':
            px = ifft2c(((1/(1+self.stepsize))*self.M+self.neg_M)*(fft2c(x)+self.stepsize*self.M*y))
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
            x = torch.clamp(x,-0,1)
            px = x
        else:
            ValueError('Forward Model degradation not implemented')
    elif self.noise_model == 'speckle': 
        # Implement a gradient descent to approximate the Prox
        input = x
        vv = input.clone()
        with torch.enable_grad():
            for _ in range(5):
                uu = vv.clone()
                uu = uu.detach().requires_grad_(True)
                norm = (0.5/stepsize)*torch.sum(torch.abs(uu - y)**2) + self.f_speckle(uu, y)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=uu)[0]
                vv -= norm_grad*0.001
                vv = torch.clamp(vv,-0,1)
        px = vv
        print(px)
        px = torch.clamp(px,-0,1)
    else :  
        ValueError('Forward Model noise model not implemented')
    return px


def genMask(imgSize, numLines, device='cpu'):
    """
    Generate a mask for MRI reconstruction in torch.
    It is a translation in torch of the code proposed in https://github.com/wustl-cig/bcred/tree/master
    
    Args:
        imgSize (tuple): (high, width) of the image; need to be a multiple of 2
        numLines (int): number of ligne to draw
        device (str): 'cpu' or 'cuda'
    
    Returns:
        torch.BoolTensor: binary mask of size imgSize
    """
    H, W = imgSize
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("image must be even sized!")
    
    center = torch.tensor([H/2 + 1, W/2 + 1], device=device)
    freqMax = np.ceil(np.sqrt((H/2)**2 + (W/2)**2))
    
    # angles of the lines
    ang = torch.linspace(0, np.pi, steps=numLines+1, device=device)[:-1]  # we remove the last one to avoid multiples
    
    mask = torch.zeros(imgSize, dtype=torch.bool, device=device)
    
    # relative coordonates
    offsets = torch.arange(-freqMax, freqMax+1, device=device)
    
    for theta in ang:
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        # Float coordonnates
        ix = center[1] + offsets * cos_t
        iy = center[0] + offsets * sin_t
        
        # Integer coordonates
        ix = torch.floor(ix + 0.5).long()
        iy = torch.floor(iy + 0.5).long()
        
        # Filter to keep the valid indexes
        valid = (ix >= 1) & (ix <= W) & (iy >= 1) & (iy <= H)
        ix = ix[valid] - 1  # to go to the 1-based index
        iy = iy[valid] - 1
        
        mask[iy, ix] = True
    
    return mask

def fft2c(x):
    """
    2D centered FFT (unitary)
    """
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            norm="ortho"
        ),
        dim=(-2, -1)
    )

def ifft2c(k):
    """
    2D centered iFFT (unitary)
    """
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(k, dim=(-2, -1)),
            norm="ortho"
        ),
        dim=(-2, -1)
    )

def injectspeckle_amplitude_log(img, L, device, gen):
    """
    Injected speckle noise in the image img with the number of looks L
    """
    img_exp = torch.exp(img)
    rows = img_exp.shape[-2]
    columns = img_exp.shape[-1]
    s = torch.zeros((rows, columns))
    for k in range(0,L):
        gamma = torch.abs( torch.randn(rows,columns, generator = gen) + torch.randn(rows,columns, generator = gen)*1j )**2/2
        s = s + gamma
    s_amplitude = torch.sqrt(s/L).to(device)
    ima_speckle_amplitude = img_exp * s_amplitude[None,None,:,:]
    ima_speckle_log = torch.log(ima_speckle_amplitude)
    return ima_speckle_log