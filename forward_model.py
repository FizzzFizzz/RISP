import torch.nn as nn
import torch
import hdf5storage
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import cv2
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
        fft_k = deblur.p2o(self.kernel, clean_image.shape[-2:]).to(self.device)
        temp = fft_k.to(torch.complex64) * torch.fft.fft2(clean_image.to(torch.complex64))
        temp = temp.contiguous()
        temp_cpu = temp.detach().cpu().clone() # to avoid some possible CUDA error in the ifft2
        obs_cpu = torch.fft.ifft2(temp_cpu)
        observation_without_noise = obs_cpu.to(temp.device)
        noise = (sigma / 255) * torch.normal(torch.zeros(observation_without_noise.size()), torch.ones(observation_without_noise.size()), generator = gen)
        noise = noise.to(self.device)
        return observation_without_noise + noise
    elif self.Pb == "SR":
        # Degrade image
        clean_image_np = np.float32(util.tensor2uint(clean_image) / 255.)
        k_np = util.tensor2uint(self.kernel)
        blur_im = utils_sr.numpy_degradation(clean_image_np, k_np, self.sf)
        noise = np.random.normal(0, 1, blur_im.shape) * sigma / 255.
        blur_im += noise
        # Initialize the algorithm
        init_im = cv2.resize(blur_im, (int(blur_im.shape[1] * self.sf), int(blur_im.shape[0] * self.sf)), interpolation=cv2.INTER_CUBIC)
        init_im = utils_sr.shift_pixel(init_im, self.sf)
        return util.uint2tensor4(blur_im).to(self.device), util.uint2tensor4(init_im).to(self.device)
    elif self.Pb == "rician":
        observation_without_noise = clean_image
        gen_real = torch.Generator()
        gen_real.manual_seed(seed)
        gen_imag = torch.Generator()
        gen_imag.manual_seed(seed*2)
        # For Rician noise, we need two orthogonal i.i.d. gaussian noises.
        noise_real = (torch.normal(torch.zeros(observation_without_noise.size()), torch.ones(observation_without_noise.size()), generator = gen_real)*sigma / 255).to(self.device)
        noise_imag = (torch.normal(torch.zeros(observation_without_noise.size()), torch.ones(observation_without_noise.size()), generator = gen_imag)*sigma / 255).to(self.device)
        observation = torch.sqrt( (observation_without_noise+noise_real)**2 + noise_imag**2)
        return observation.to(self.device)
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
        pseudo_inverse = torch.real(ifft2c(M*observation.clone()))
        return observation, M, pseudo_inverse
    elif self.Pb == "speckle":
        observation = injectspeckle_amplitude_log(clean_image, L = self.L, device = self.device, gen = gen)
        return observation
    else:
        raise ValueError("Forward Model not implemented")


def data_fidelity_init(self, init = None):
    if self.Pb == 'deblurring':
        # Initialization of Gradient operator
        fft_k = deblur.p2o(self.kernel, self.observation.shape[-2:])
        self.fft_k = fft_k
        fft_kH = torch.conj(fft_k)
        self.fft_kH = fft_kH
        abs_k = fft_kH * fft_k
        self.abs_k = abs_k
    elif self.Pb == 'SR':
        # Initialization of Proximal operator
        self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(self.observation, self.kernel, self.sf)
    elif self.Pb == 'inpainting':
        self.M = (self.mask).clone()
    elif self.Pb == "MRI":
        self.neg_M = torch.ones(self.M.shape).to(self.device) - 1*self.M
    elif self.Pb == 'ODT':
        from utils.inverse_scatter import InverseScatter
        shape0 = init.shape[2]
        shape1 = init.shape[3]
        shape2 = self.obs.shape[2]
        shape3 = self.obs.shape[1]
        self.scatter_op = InverseScatter(
            Lx=0.18, Ly=0.18, Nx=shape0, Ny=shape1, 
            wave=6, numRec=shape2, numTrans=shape3,
            sensorRadius=1.6, svd=False
        )
    elif self.Pb == "speckle":
        self.f_speckle = lambda u, y : torch.sum(self.L * (u + torch.exp(y - u)))
    elif self.Pb == "rician":
        print('Rician noise removal')
    else:
        raise ValueError("Forward Model not implemented")


def A(self, x):
    """
    Calculation A*x with A the linear degradation operator 
    """
    if self.Pb == 'deblurring':
        Ax = torch.real(deblur.ifftn(self.fft_k * deblur.fftn(x)))
    elif self.Pb == 'SR':
        Ax = utils_sr.G(x, self.kernel, sf=self.sf)
    elif self.Pb == 'inpainting':
        Ax = self.M * x
    elif self.Pb == 'MRI':
        Ax = self.M[None,None,:,:] * fft2c(x)
    elif self.Pb == 'despeckle':
        Ax = x
    elif self.Pb == 'rician':
        Ax = x
    else:
        raise ValueError('degradation not implemented')
    return Ax  


def compute_data_fidelity(self, x, obs):
    """
    A function to compute the data-fidelity at point x with the observation obs.
    """
    Ax = A(self, x)
    if self.noise_model == 'gaussian':
        f = 0.5 * torch.norm(obs - Ax, p=2) ** 2
    elif self.noise_model == 'speckle':
        f = self.L*(Ax + torch.exp(obs - Ax)).sum()
    return f

def compute_data_grad(self, x, obs):
    if self.noise_model == 'gaussian':
        if self.Pb == 'deblurring':
            data_grad = self.abs_k * deblur.fftn(x) - self.fft_kH * deblur.fftn(obs)
            data_grad = torch.real(deblur.ifftn(data_grad))
        elif self.Pb == "SR":
            data_grad = utils_sr.grad_solution_L2(x, obs, self.kernel, self.sf)
        elif self.Pb == 'inpainting':
            data_grad = self.M * (x - obs)
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
    elif self.noise_model == 'rician':
        if self.Pb == 'rician':
            temp_sig = self.sigma_obs/255
            data_grad = x / temp_sig**2 - (obs / temp_sig**2) * SpecialB((x*obs)/temp_sig**2)
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
        if self.Pb == 'deblurring':
            num = (stepsize * self.fft_kH * deblur.fftn(y) + deblur.fftn(x))
            den = 1 + stepsize * self.abs_k
            px_fourier = num / den
            px = torch.real(deblur.ifftn(px_fourier))
        elif self.Pb == 'SR':
            px = utils_sr.prox_solution_L2(x, self.FB, self.FBC, self.F2B, self.FBFy, stepsize, self.sf)
        elif self.Pb == 'inpainting':
            if self.sigma_obs > 1e-2:
                px = (stepsize*self.M*y + x)/(stepsize*self.M+1)
            else :
                px = self.M*y + (1-self.M)*x
        elif self.Pb == 'MRI':
            px = ifft2c(((1/(1+stepsize))*self.M+self.neg_M)*(fft2c(x)+stepsize*self.M*y))
        elif self.Pb == 'ODT':
            input = x
            uu = input.clone()
            vv = input.clone()
            with torch.enable_grad():
                tt = 0
                crit = 1
                dt = 400/stepsize / self.lamb
                while (tt<100)&(crit>2e-3):
                    tt += 1
                    uu = vv.clone()
                    uu = uu.detach().requires_grad_(True)
                    uscat_pred = self.scatter_op.forward(uu, unnormalize=False)
                    difference = uscat_pred - y
                    norm = 0.5*(stepsize*self.lamb)*torch.sum(torch.abs(difference)**2) + 0.5*torch.sum(torch.abs(uu-input)**2)
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=uu)[0]
                    data_grad = norm_grad
                    vv -= dt*data_grad
                    crit = torch.sum(torch.abs(data_grad.detach())**2)
            x = uu
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
    elif self.noise_model == 'rician': 
        px = _fast_irl1(y, x, self.sigma_obs, stepsize, irl1_iter_num=15)
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

def SpecialB(x):
    return torch.special.i1e(x) / torch.special.i0e(x)


def _fast_irl1(obs, input, sigma, lamb, irl1_iter_num=10):
    f = obs
    v = input
    sigma = sigma/255
    irl1_input = input
    f_sigma2 = f / sigma**2
    lamb_f_sigma2 = lamb * f_sigma2
    lamb_sigma2_beta = lamb/sigma**2 + 1
    for _ in range(irl1_iter_num):
        Iz = SpecialB(f_sigma2 * v)
        Iz = torch.clamp(Iz, min=0)
        y = f_sigma2 * (1 - Iz)
        v = (lamb_f_sigma2 + irl1_input - lamb*y) / lamb_sigma2_beta
    return v
