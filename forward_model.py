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



def data_fidelity_init(self, kernel = None, init = None):
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
        self.M = array2tensor(degradation).to(self.device)
    else:
        raise ValueError("Forward Model not implemented")


def compute_data_grad(self, x, obs):
    if self.Pb == 'deblurring':
        data_grad = self.abs_k * deblur.fftn(x) - self.fft_kH * deblur.fftn(obs)
        data_grad = torch.real(deblur.ifftn(data_grad))
        return data_grad
    else:
        raise ValueError("Forward Model not implemented")

# def data_fidelity_grad(self, x, y):
#     """
#     Calculate the gradient of the data-fidelity term.
#     :param x: Point where to evaluate F
#     :param y: Degraded image
#     """
#     if self.hparams.noise_model == 'gaussian':
#         if self.hparams.degradation_mode == 'deblurring':
#             return utils_sr.grad_solution_L2(x, y, self.k_tensor, self.sf)
#         elif self.hparams.degradation_mode == 'inpainting':
#             return 2 * self.M * (x - y)
#         else:
#             raise ValueError('degradation not implemented')
#     elif self.hparams.noise_model == 'speckle':
#         return self.hparams.L*(1-torch.exp(y-x))
#     else:
#         raise ValueError('noise model not implemented')


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
        else:
            ValueError('Forward Model degradation not implemented')
    else :  
        ValueError('Forward Model noise model not implemented')
    return px

