import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from lpips import LPIPS
from argparse import ArgumentParser
import os
import argparse
import cv2
import imageio


parser = ArgumentParser()
parser.add_argument('--fig_number', type=int, default=-1)
parser.add_argument('--table_number', type=int, default=-1)
pars = parser.parse_args()

path_figure = "results/figure/"


if pars.fig_number == 0:
    #generate figure for inpainting qualitative comparison in the paper on the castle image.
    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_"

    theta_list = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    fig = plt.figure()
    
    for theta in theta_list:
        psnr_list = []
        for i in range(5):
            dic = np.load(path_result + str(theta) + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = str(theta))
    
    path_result = "results/set5/RED_level_0.1_lamb18"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')

    plt.legend()
    fig.savefig(path_figure+'/result_momentum_various_theta.png', dpi = 300)
    plt.show()