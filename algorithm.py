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
import deepinv
import utils_image as util
import utils_logger
import utils_deblur as deblur
from argparse import ArgumentParser
from denoiser_model import *
from forward_model import *


class PnP(nn.Module):
    def __init__(self, nb_itr=50, denoiser_name = "GSDRUNet", n_channels = 3, device = 'cpu', Pb = 'deblurring', sigma_obs = 0, noise_model = "gaussian", sf = 1):
        '''
            nb_itr : number of iterations of the PnP algorithm
        '''
        super(PnP, self).__init__()
        self.nb_itr = nb_itr
        self.device = device
        if denoiser_name == "DRUNet":
            denoiser_net = Drunet_running(model_path = "models_ckpt/drunet_color.pth", n_channels = n_channels, device = device)
        elif denoiser_name == "GSDRUNet":
            # The pretrained GSDRNet weights can be download in : https://huggingface.co/deepinv/gradientstep/blob/main/GSDRUNet.ckpt
            denoiser_net = deepinv.models.GSDRUNet(pretrained = "models_ckpt/GSDRUNet.ckpt", device = device)
        elif denoiser_name == "GSDRUNet_SoftPlus":
            # The pretrained GSDRNet weights can be download in : https://plmbox.math.cnrs.fr/f/414fbb3e754840978ef8/?dl=1
            denoiser_net = deepinv.models.GSDRUNet(pretrained = "models_ckpt/GSDRUNet_SoftPlus.ckpt", act_mode = "s", device = device)
        elif denoiser_name == "GSDRUNet_grayscale":
            # The pretrained GSDRUnet weights in grayscale can be download in : https://huggingface.co/deepinv/gradientstep/resolve/main/GSDRUNet_grayscale_torch.ckpt
            denoiser_net = deepinv.models.GSDRUNet(pretrained = "models_ckpt/GSDRUNet_grayscale.ckpt", in_channels=1, out_channels=1, device = device)
        else:
            raise ValueError("Denoiser not implemented.")
        self.Pb = Pb
        self.sigma_obs = sigma_obs
        self.noise_model = noise_model
        self.sf = 1
        self.net = denoiser_net

        # only test
        self.res = {}
        self.res['psnr'] = [0] * (nb_itr + 1)
        self.res['ssim'] = [0] * (nb_itr + 1)
        self.res['image'] = [0] * (nb_itr + 1)
        self.nb_restart_activ = 0
        

    def get_psnr_i(self, u, clean, i):
        '''
        Compute the PSNR, SSIM and save the image at the iteration i.
        '''
        psnr = util.calculate_psnr_torch(u, clean).item()
        self.res['psnr'][i] = psnr
        ssim = util.calculate_ssim_torch(u, clean).item()
        self.res['ssim'][i] = ssim
        pre_i = torch.clamp(u, 0., 1.)
        self.res['image'][i] = ToPILImage()(pre_i[0])

    def forward(self, initial_uv, obs, clean, sigma_obs, lamb=690, denoiser_sigma=25./255., theta = 0.9, r=3, B = 5000., Nesterov = False, momentum = False, restarting_su = False, restarting_li = False, stepsize = 0.02, alg = "GD",adapative_restart = False,adapative_restart_factor = 0.5):
        '''
            Computed the RED Algorithm with
                initial_uv : the initialization for the algorithm
                obs : the observation, degraded image
                clean : the clean image to compute the metrics at each iterations
                sigma_obs : the noise level of the observation
                lamb : the regularization parameter, multiplicative factor in front of the data-fidelity
                denoiser_sigma : the noise parameter of the denoiser
                theta : momentum parameter for fixed momentum
                r : the parameter of the Generalized Nesterov momentum, r = 3, we recover the classic Nesterov momentum
                B : parameter for the Li restarting criterion
                Nesterov : boolean which gives if the Generalize Nesterov momentum is used
                momentum : boolean which gives if fixed momentum is used
                restarting_su : boolean which gives if the restarting criterion of Su is used (for Generalize momentum)
                restarting_li : boolean which gives if the restarting criterion of Li is used (for fixed momentum)
                stepsize : the stepsize of the algorithm
                alg : Type of algorithm that is computed. If alg == "GD", a full gradient is computed, 
                        if alg == "PGD", a gradient is computed on the regularization and a proximal-step on the data-fidelity
        '''
        # init
        u  = initial_uv
        data_fidelity_init(self, init = initial_uv) # Initialize the data-fidelity operators
        self.nb_restart_activ = 0

        self.lamb = lamb
        y_denoised = u
        x = u
        y = u

        out = y_denoised
        
        if restarting_su:
            # Restarting criterion proposed by "A Differential Equation for Modeling Nesterov’s Accelerated Gradient Method: Theory and Insights" section 5
            restart_crit_su = -float('inf') 
            j = 0
            k_min = 5
        if restarting_li:
            #Restarting criterion proposed by "Restarted Nonconvex Accelerated Gradient Descent: No More Polylogarithmic Factor in the O( 7 4) Complexity"
            restart_crit_li = 0
            j = 0

        for k in tqdm(range(self.nb_itr)):
            x_old = x
            self.get_psnr_i(torch.clamp(torch.real(out), min = -0., max =1.), clean, k)

            if self.Pb == "inpainting" and k < 10:
                sigma_den = 50. / 255.
            else:
                sigma_den = denoiser_sigma

            data_grad = compute_data_grad(self, y, obs)
            y = y.type(torch.cuda.FloatTensor)
            y_denoised = self.net.forward(y, sigma_den)
            reg_grad = y - y_denoised

            if alg == "GD":
                # print("reg")
                # print(torch.sqrt(torch.sum(torch.abs(reg_grad)**2)))
                # print("data")
                # print(torch.sqrt(torch.sum(torch.abs(data_grad)**2)))
                grad = reg_grad + lamb * data_grad
                x = y - stepsize*grad
                out = y_denoised
                if self.Pb == 'ODT':
                    x = torch.clamp(x,-0.,1.)
                    out = x
            elif alg == "PGD":
                if self.Pb == 'ODT':
                    x = data_fidelity_prox_step(self, y - stepsize*reg_grad, obs, stepsize)
                    out = x
                else: 
                    x = data_fidelity_prox_step(self, y - (stepsize/lamb)*reg_grad, obs, stepsize)
                    out = y_denoised

            if restarting_su:
                restart_crit_su_old = restart_crit_su
                restart_crit_su = torch.mean(torch.abs(x-x_old)).item()
                if (k>k_min) and (restart_crit_su<restart_crit_su_old):
                    j = 0
                    x_old = x
                    self.nb_restart_activ += 1
                    restart_crit_su = -float('inf')
                else:
                    j += 1
            elif restarting_li:
                restart_crit_li = restart_crit_li + torch.sum((x-x_old)**2).item()
                if j * restart_crit_li > B:
                    j = 0
                    if adapative_restart == True:
                        B *= adapative_restart_factor
                    x_old = x
                    self.nb_restart_activ += 1
                    restart_crit_li = 0
                else:
                    j += 1
            else:
                j = k

            if Nesterov:
                if k+r==0:
                    y = x + (x-x_old)
                else:
                    y = x + j/(j+r)*(x-x_old)
            elif momentum:
                y = x + (1-theta)*(x - x_old)
            else:
                y = x
        self.get_psnr_i(torch.clamp(torch.real(out), min = -0., max =1.), clean, self.nb_itr) # put the last iterate at the end of the stack