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
import sys 
sys.path.append("utils/")
import utils_image as util


parser = ArgumentParser()
parser.add_argument('--fig_number', type=int, default=-1)
parser.add_argument('--prep_fig', type=int, default=-1)
parser.add_argument('--table_number', type=int, default=-1)
pars = parser.parse_args()

path_figure = "results/figure/"


if pars.fig_number == 0:
    # Generate figure for deblurring on 10 images with various momentum parameter with a motion blur with restarting for GD
    path_result = "results/deblurring/CBSD10/"

    theta_list = [0.01, 0.1, 0.2, 0.3, 0.9]
    fig = plt.figure()
    
    for theta in theta_list:
        psnr_list = []
        for i in range(10):
            dic = np.load(path_result + "GD/kernel_0/Momentum/theta_" + str(theta) + "/restarting_li/den_level_0.1/lamb_15.0/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        psnr_std = np.std(psnr_list, axis = 0)
        psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$\theta = $"+str(theta))
        plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    
    path_result = "results/deblurring/CBSD10/GD/kernel_0/den_level_0.1/lamb_15.0/stepsize_0.1"
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    psnr_std = np.std(psnr_list, axis = 0)
    psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
    psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
    line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0,49], ["0","50"])
    plt.ylim(15,30)
    plt.yticks([15,30], ["15","30"])
    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ on $10$ images" +"\n" + r"from CBSD68 dataset with various momentum parameter $\theta$")
    fig.savefig(path_figure+'result_momentum_various_theta_GD.png', dpi = 300)
    plt.show()



if pars.fig_number == 1:
    # Generate figure for deblurring on 10 images with various momentum parameter with a motion blur without restarting for GD
    path_result = "results/deblurring/CBSD10/"

    theta_list = [0.01, 0.1, 0.2, 0.3, 0.9]
    fig = plt.figure()
    
    for theta in theta_list:
        psnr_list = []
        for i in range(10):
            dic = np.load(path_result + "GD/kernel_0/Momentum/theta_" + str(theta) + "/den_level_0.1/lamb_15.0/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        psnr_std = np.std(psnr_list, axis = 0)
        psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$\theta = $"+str(theta))
        plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    
    path_result = "results/deblurring/CBSD10/GD/kernel_0/den_level_0.1/lamb_15.0/stepsize_0.1"
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    psnr_std = np.std(psnr_list, axis = 0)
    psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
    psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
    line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0,49], ["0","50"])
    plt.ylim(15,30)
    plt.yticks([15,30], ["15","30"])
    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ on $10$ images" +"\n" + r"from CBSD68 dataset with various momentum parameter $\theta$")
    fig.savefig(path_figure+'result_momentum_various_theta_GD_without_restart.png', dpi = 300)
    plt.show()


if pars.prep_fig == 2:
    path_result = "results/deblurring/CBSD10/"
    method_name = ["RED", r"RiRED $\theta = 0.2$", "Prox-RED", r"Prox-RiRED $\theta = 0.2$"]
    stepsize_method = [0.1, 0.07, 2.0, 5.0]
    nb_method = len(method_name)
    psnr_list = [[] for _ in range(nb_method)]
    residuals = [[] for _ in range(nb_method)]
    f_list = [[] for _ in range(nb_method)]
    g_list = [[] for _ in range(nb_method)]
    F_list = [[] for _ in range(nb_method)]
    nabla_F_list = [[] for _ in range(nb_method)]

    for k in tqdm(range(10)):
        path_method = ["GD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/", "GD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/Momentum/theta_0.2/restarting_li/", "PGD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/", "PGD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/Momentum/theta_0.2/restarting_li/"]
        for i in range(10):
            for j in range(nb_method):
                stepsize = str(stepsize_method[j])
                dic = np.load(path_result + path_method[j] + "/den_level_0.1/lamb_15.0/stepsize_"+stepsize+"/dict_results_"+str(i)+".npy", allow_pickle=True).item()
                psnr_list[j].append(dic['psnr_list'])
                F_list[j].append(dic['F_list'])
                f_list[j].append(dic['f_list'])
                g_list[j].append(dic['g_list'])
                nabla_F_list[j].append(dic['nabla_F_list'])
                
                stack_im = dic['stack_images']
                ref = np.sum(np.array(stack_im[0])**2)
                residuals_stack = []
                for l in range(len(stack_im) - 1):
                    residuals_stack.append(np.sum((np.array(stack_im[l+1]) - np.array(stack_im[l]))**2) / ref)
                residuals[j].append(residuals_stack)

    psnr_list = np.array(psnr_list)
    f_list = np.array(f_list)
    g_list = np.array(g_list)
    F_list = np.array(F_list)
    nabla_F_list = np.array(nabla_F_list)
    residuals = np.array(residuals)

    dict = {
        'psnr_list' : psnr_list,
        'f_list' : f_list,
        'g_list' : g_list,
        'F_list' : F_list,
        'nabla_F_list' : F_list,
        'nabla_F_list' : nabla_F_list,
        'method_name' : method_name,
        'nb_method' : nb_method,
        'stepsize_method' : stepsize_method,
        'residuals' : residuals,
        }
    np.save(path_figure+"/result_deblurring_expe", dict)


if pars.fig_number == 2:
    # Generate figure for convergence of the method for deblurring
    dic = np.load(path_figure+"/result_deblurring_expe.npy", allow_pickle=True).item()
    
    psnr_list = dic['psnr_list']
    f_list = dic['f_list']
    g_list = dic['g_list']
    F_list = dic['F_list']
    nabla_F_list = dic['nabla_F_list']
    nb_method = dic['nb_method']
    method_name = dic['method_name']
    residuals = dic['residuals']

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_indx = [3, 1, 2, 0]
    fig = plt.figure()

    for j in range(nb_method):
        psnr_mean = np.mean(psnr_list[j], axis = 0)
        convergence_time = np.sum(psnr_mean < 27.7)
        psnr_std = np.std(psnr_list[j], axis = 0)
        psnr_min = np.quantile(psnr_list[j], 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = method_name[j], color = colors[color_indx[j]])
        # plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    size_number = 15

    # plt.plot([0, 50], [27.7, 27.7], linestyle = "--", color = 'gray')

    plt.xlim(0,49)
    plt.xticks([0, 9, 20, 38, 49], ["0", "9", "20", "38", "50"], fontsize = size_number)
    plt.ylim(18,29)
    plt.yticks([18, 29], ["18", "29"], fontsize = size_number)
    # plt.legend()
    plt.title(r"Convergence PSNR for deblurring")
    fig.savefig(path_figure+'convergence_PSNR_deblurring.png', dpi = 300)
    plt.show()


    fig = plt.figure()
    
    for j in range(nb_method):
        residuals_mean = np.mean(np.log10(residuals[j]), axis = 0)
        residuals_min = np.quantile(np.log10(residuals[j]), 0.25, axis = 0)
        residuals_max = np.quantile(np.log10(residuals[j]), 0.75, axis = 0)
        line, = plt.plot(np.arange(len(residuals_mean)), residuals_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(residuals_mean)), residuals_min, residuals_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    ticks = np.arange(-5, 1, 1)
    labels = [r"$10^{-5}$" if i == 0 else
            r"$10^0$" if i == len(ticks) - 1 else ""
            for i in range(len(ticks))]
    plt.yticks(ticks, labels, fontsize = size_number)
    plt.ylabel(r"$\|x^{k+1} - x^k\|^2 / \|x^0\|^2$", labelpad=-10, fontsize = size_number)

    # plt.legend()
    plt.title(r"Convergence residuals for deblurring")
    fig.savefig(path_figure+'convergence_residuals_deblurring.png', dpi = 300)
    plt.show()

    fig = plt.figure()
    
    for j in range(nb_method):
        f_list_mean = np.mean(f_list[j], axis = 0)
        f_list_min = np.quantile(f_list[j], 0.25, axis = 0)
        f_list_max = np.quantile(f_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(f_list_mean)), f_list_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(f_list_mean)), f_list_min, f_list_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    plt.ylabel(r"$f$", fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence f for deblurring")
    fig.savefig(path_figure+'convergence_f_deblurring.png', dpi = 300)
    plt.show()



    fig = plt.figure()
    
    for j in range(nb_method):
        g_list_mean = np.mean(g_list[j], axis = 0)
        g_list_min = np.quantile(g_list[j], 0.25, axis = 0)
        g_list_max = np.quantile(g_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(g_list_mean)), g_list_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(g_list_mean)), g_list_min, g_list_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    plt.ylabel(r"$g$", fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence g for deblurring")
    fig.savefig(path_figure+'convergence_g_deblurring.png', dpi = 300)
    plt.show()



    fig = plt.figure()
    
    for j in range(nb_method):
        F_list_mean = np.mean(F_list[j], axis = 0)
        F_list_min = np.quantile(F_list[j], 0.25, axis = 0)
        F_list_max = np.quantile(F_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(F_list_mean)), F_list_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(F_list_mean)), F_list_min, F_list_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    plt.ylabel(r"$F = \lambda f + g$", fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence F for deblurring")
    fig.savefig(path_figure+'convergence_F_deblurring.png', dpi = 300)
    plt.show()

    fig = plt.figure()
    
    for j in range(nb_method):
        nabla_F_list_mean = np.mean(nabla_F_list[j], axis = 0)
        nabla_F_list_min = np.quantile(nabla_F_list[j], 0.25, axis = 0)
        nabla_F_list_max = np.quantile(nabla_F_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(nabla_F_list_mean)), nabla_F_list_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(nabla_F_list_mean)), nabla_F_list_min, nabla_F_list_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    plt.ylabel(r"$\|\nabla F(x^k)\|$", fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence nabla F for deblurring")
    fig.savefig(path_figure+'convergence_nabla_F_deblurring.png', dpi = 300)
    plt.show()


if pars.prep_fig == 3:
    #generate figure for deblurring on 10 images with 10 kernels of blur with various momentum parameter for GD
    path_result = "results/deblurring/CBSD10/GD/kernel_"

    theta_list = [0.01, 0.1, 0.2, 0.3, 0.9]
    psnr_list_without_restart = [[] for _ in range(len(theta_list) + 1)]
    psnr_list_with_restart = [[] for _ in range(len(theta_list) + 1)]
    
    for k in tqdm(range(10)):
        for i in range(10):
            for j, theta in enumerate(theta_list):
                dic = np.load(path_result + str(k) + "/Momentum/theta_" + str(theta) + "/den_level_0.1/lamb_15.0/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
                psnr_list_without_restart[j].append(dic['psnr_list'])
                dic = np.load(path_result + str(k) + "/Momentum/theta_" + str(theta) + "/restarting_li/den_level_0.1/lamb_15.0/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
                psnr_list_with_restart[j].append(dic['psnr_list'])
            dic = np.load(path_result + str(k) + "/den_level_0.1/lamb_15.0/stepsize_0.1/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list_without_restart[-1].append(dic['psnr_list'])
            psnr_list_with_restart[-1].append(dic['psnr_list'])
    psnr_list_without_restart = np.array(psnr_list_without_restart)
    psnr_list_with_restart = np.array(psnr_list_with_restart)
    dict= {
        'theta_list' : theta_list,
        'psnr_list_without_restart' : psnr_list_without_restart,
        'psnr_list_with_restart' : psnr_list_with_restart,
    }
    np.save(path_figure+"/result_deblurring_various_theta", dict)


if pars.fig_number == 3:
    #generate figure for deblurring on 10 images with 10 kernels of blur with various momentum parameter for GD
    dic = np.load(path_figure+"/result_deblurring_various_theta.npy", allow_pickle=True).item()
    theta_list = dic['theta_list']
    psnr_list_without_restart = dic['psnr_list_without_restart']
    psnr_list_with_restart = dic['psnr_list_with_restart']
       

    fig = plt.figure()
    for i, theta in enumerate(theta_list):
        psnr_mean = np.mean(psnr_list_without_restart[i], axis = 0)
        psnr_min = np.quantile(psnr_list_without_restart[i], 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list_without_restart[i], 0.75, axis = 0)
        line,  = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$\theta = $"+str(theta))
        plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    psnr_mean = np.mean(psnr_list_without_restart[-1], axis = 0)
    psnr_min = np.quantile(psnr_list_without_restart[-1], 0.25, axis = 0)
    psnr_max = np.quantile(psnr_list_without_restart[-1], 0.75, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"RED")
    plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    size_number = 15
    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)
    plt.ylim(18,29)
    plt.yticks([18, 28], ["18", "28"], fontsize = size_number)
    plt.legend()
    fig.savefig(path_figure+'/result_deblurring_various_theta_without_restart.png', dpi = 300)

    fig = plt.figure()
    for i, theta in enumerate(theta_list):
        psnr_mean = np.mean(psnr_list_with_restart[i], axis = 0)
        psnr_min = np.quantile(psnr_list_with_restart[i], 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list_with_restart[i], 0.75, axis = 0)
        line,  = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$\theta = $"+str(theta))
        plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    psnr_mean = np.mean(psnr_list_with_restart[-1], axis = 0)
    psnr_min = np.quantile(psnr_list_with_restart[-1], 0.25, axis = 0)
    psnr_max = np.quantile(psnr_list_with_restart[-1], 0.75, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"RED")
    plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)
    plt.ylim(18,29)
    plt.yticks([18, 28], ["18", "28"], fontsize = size_number)
    plt.legend()
    fig.savefig(path_figure+'/result_deblurring_various_theta_with_restart.png', dpi = 300)



if pars.fig_number == 4:
    # Generate a figure to show various result on image deblurring with various kernels and different time for each method
    path_result = "results/deblurring/CBSD10/"

    n = 7
    m = 6

    size_title = 40
    size_label = 25

    #size of the black rectangle
    height = 30
    width = 210

    fig = plt.figure(figsize = (m*7.44, n*5))
    gs = gridspec.GridSpec(n, m, hspace = 0, wspace = 0)
    kernel_list = [0,1,2,3,4,9,8,7,6,5]
    for i in tqdm(range(n)):
        dic_RED = np.load(path_result + "GD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(kernel_list[i])+"/den_level_0.1/lamb_15.0/stepsize_0.1/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        im = dic_RED['stack_images'][-1]
        im.save(path_figure+'images_deblurring/im_'+str(i)+'_restored_50_itr_RED_psnr_'+str(dic_RED['psnr_restored'])[:4]+'.png')

        dic_ProxRED = np.load(path_result + "PGD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(kernel_list[i])+"/den_level_0.1/lamb_15.0/stepsize_2.0/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        im = dic_ProxRED['stack_images'][-1]
        im.save(path_figure+'images_deblurring/im_'+str(i)+'_restored_38_itr_Prox_RED_psnr_'+str(dic_ProxRED['psnr_restored'])[:4]+'.png')
        
        dic_RiRED = np.load(path_result + "GD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(kernel_list[i])+"/Momentum/theta_0.2/restarting_li/den_level_0.1/lamb_15.0/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        im = dic_RiRED['stack_images'][20]
        im.save(path_figure+'images_deblurring/im_'+str(i)+'_restored_20_itr_RISP_psnr_'+str(dic_RiRED['psnr_list'][20])[:4]+'.png')

        dic_ProxRiRED = np.load(path_result + "PGD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(kernel_list[i])+"/Momentum/theta_0.2/restarting_li/den_level_0.1/lamb_15.0/stepsize_5.0/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        im = dic_ProxRiRED['stack_images'][9]
        im.save(path_figure+'images_deblurring/im_'+str(i)+'_restored_9_itr_Prox_RISP_psnr_'+str(dic_ProxRiRED['psnr_list'][9])[:4]+'.png')

        GT = np.array(dic_RED["clean_image"]).astype(np.uint8)
        plt.imsave(path_figure+'images_deblurring/im_'+str(i)+'_GT.png', GT)

        kern = dic_RED["kernel"]
        plt.imsave(path_figure+'images_deblurring/im_'+str(i)+'_kernel.png', kern, cmap = 'gray')

        obs = np.array(dic_RED["observation"]); gt = np.array(dic_RED["clean_image"])
        plt.imsave(path_figure+'images_deblurring/im_'+str(i)+'_obs_psnr_'+str(PSNR(obs, gt))[:4]+'.png', obs.astype(np.uint8))



if pars.fig_number == 5:
    #generate figure for deblurring on 10 images with various restarting parameter and the momentum parameter theta = 0.2 or theta = 0.01
    path_result = "results/deblurring/CBSD10/GD/denoiser_name_GSDRUNet_SoftPlus/kernel_0/Momentum/theta_0.2/"

    B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
    fig = plt.figure()
    
    for B in B_list:
        psnr_list = []
        for i in range(10):
            dic = np.load(path_result + 'restarting_li/den_level_0.1/lamb_15.0/B_' + str(B) + "/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "den_level_0.1/lamb_15.0/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = +\infty$")
    

    path_result = "results/deblurring/CBSD10/GD/denoiser_name_GSDRUNet_SoftPlus/kernel_0/"
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "den_level_0.1/lamb_15.0/stepsize_0.1/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED', linestyle='--')

    plt.legend()
    fig.savefig(path_figure+'/result_momentum_various_B_theta_0.2.png', dpi = 300)
    plt.show()

    # For theta = 0.01

    path_result = "results/deblurring/CBSD10/GD/denoiser_name_GSDRUNet_SoftPlus/kernel_0/Momentum/theta_0.01/"

    B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
    fig = plt.figure()
    
    for B in B_list:
        psnr_list = []
        for i in range(10):
            dic = np.load(path_result + 'restarting_li/den_level_0.1/lamb_15.0/B_' + str(B) + "/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "den_level_0.1/lamb_15.0/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = +\infty$")
    

    path_result = "results/deblurring/CBSD10/GD/denoiser_name_GSDRUNet_SoftPlus/kernel_0/"
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "den_level_0.1/lamb_15.0/stepsize_0.1/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED', linestyle='--')

    plt.legend()
    fig.savefig(path_figure+'/result_momentum_various_B_theta_0.01.png', dpi = 300)
    plt.show()



if pars.prep_fig == 6:
    # Preparate the plot of the convergence results of various methods for inpainting
    path_result = "results/inpainting/CBSD68/p_0.2/"

    method_name = ["RED", r"RiRED $\theta = 0.2$", "Prox-RED", r"Prox-RiRED $\theta = 0.2$"]
    stepsize_method = [0.1, 0.1, 5.0, 5.0]
    nb_method = len(method_name)
    psnr_list = [[] for _ in range(nb_method)]
    residuals = [[] for _ in range(nb_method)]
    F_list = [[] for _ in range(nb_method)]
    f_list = [[] for _ in range(nb_method)]
    g_list = [[] for _ in range(nb_method)]
    nabla_F_list = [[] for _ in range(nb_method)]

    path_method = ["GD/sigma_obs_1.0/denoiser_name_GSDRUNet_SoftPlus/", "GD/sigma_obs_1.0/denoiser_name_GSDRUNet_SoftPlus/Momentum/theta_0.2/restarting_li/", "PGD/sigma_obs_1.0/denoiser_name_GSDRUNet_SoftPlus/", "PGD/sigma_obs_1.0/denoiser_name_GSDRUNet_SoftPlus/Momentum/theta_0.2/restarting_li/"]
    for i in tqdm(range(10)):
        for j in range(nb_method):
            stepsize = str(stepsize_method[j])
            dic = np.load(path_result + path_method[j] + "/den_level_0.08/lamb_5.0/nb_itr_1500/stepsize_"+stepsize+"/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list[j].append(dic['psnr_list'])
            F_list[j].append(dic['F_list'])
            f_list[j].append(dic['f_list'])
            g_list[j].append(dic['g_list'])
            nabla_F_list[j].append(dic['nabla_F_list'])
            
            stack_im = dic['stack_images']
            ref = np.sum(np.array(stack_im[0])**2)
            residuals_stack = []
            for l in range(len(stack_im) - 1):
                residuals_stack.append(np.sum((np.array(stack_im[l+1]) - np.array(stack_im[l]))**2) / ref)
            residuals[j].append(residuals_stack)

    psnr_list = np.array(psnr_list)
    f_list = np.array(f_list)
    g_list = np.array(g_list)
    F_list = np.array(F_list)
    nabla_F_list = np.array(nabla_F_list)
    residuals = np.array(residuals)

    dict = {
        'psnr_list' : psnr_list,
        'f_list' : f_list,
        'g_list' : g_list,
        'F_list' : F_list,
        'nabla_F_list' : F_list,
        'nabla_F_list' : nabla_F_list,
        'method_name' : method_name,
        'nb_method' : nb_method,
        'stepsize_method' : stepsize_method,
        'residuals' : residuals,
        }
    np.save(path_figure+"/result_inpainting_expe", dict)


if pars.fig_number == 6:
    # Generate figure for convergence of various methods for inpainting
    dic = np.load(path_figure+"/result_inpainting_expe.npy", allow_pickle=True).item()
    
    psnr_list = dic['psnr_list']
    nb_method = dic['nb_method']
    method_name = dic['method_name']
    residuals = dic['residuals']
    f_list = dic['f_list']
    g_list = dic['g_list']
    F_list = dic['F_list']
    nabla_F_list = dic['nabla_F_list']

    fig = plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_indx = [3, 1, 2, 0]

    for j in range(nb_method):
        psnr_mean = np.mean(psnr_list[j], axis = 0)
        convergence_time = np.sum(psnr_mean < 27.7)
        psnr_std = np.std(psnr_list[j], axis = 0)
        psnr_min = np.quantile(psnr_list[j], 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = method_name[j], color = colors[color_indx[j]])
        # plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    size_number = 17

    # plt.plot([0, 50], [27.7, 27.7], linestyle = "--", color = 'gray')

    plt.xlim(0,1499)
    plt.xticks([0, 1499], ["0", "1500"], fontsize = size_number)
    plt.ylim(13,27)
    plt.yticks([13,27], ["13", "27"], fontsize = size_number)
    # plt.legend()
    # plt.title(r"Convergence PSNR for inpainting")
    fig.savefig(path_figure+'convergence_PSNR_inpainting.png', dpi = 300)
    plt.show()


    fig = plt.figure()
    
    for j in range(nb_method):
        residuals_mean = np.mean(np.log10(residuals[j]), axis = 0)
        residuals_min = np.quantile(np.log10(residuals[j]), 0.25, axis = 0)
        residuals_max = np.quantile(np.log10(residuals[j]), 0.75, axis = 0)
        line, = plt.plot(np.arange(len(residuals_mean)), residuals_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(residuals_mean)), residuals_min, residuals_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,1499)
    plt.xticks([0, 1499], ["0", "1500"], fontsize = size_number)

    # ticks = np.arange(-5, 1, 1)
    # labels = [r"$10^{-5}$" if i == 0 else
    #         r"$10^0$" if i == len(ticks) - 1 else ""
    #         for i in range(len(ticks))]
    # plt.yticks(ticks, labels, fontsize = size_number)
    plt.ylabel(r"$\|x^{k+1} - x^k\|^2 / \|x^0\|^2$", labelpad=-10, fontsize = size_number)

    # plt.legend()
    # plt.title(r"Convergence residuals for inpainting")
    fig.savefig(path_figure+'convergence_residuals_inpainting.png', dpi = 300)
    plt.show()


    fig = plt.figure()
    
    for j in range(nb_method):
        f_list_mean = np.mean(f_list[j], axis = 0)
        f_list_min = np.quantile(f_list[j], 0.25, axis = 0)
        f_list_max = np.quantile(f_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(f_list_mean)), f_list_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(f_list_mean)), f_list_min, f_list_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    plt.ylabel(r"$f$", fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence f for inpainting")
    fig.savefig(path_figure+'convergence_f_inpainting.png', dpi = 300)
    plt.show()



    fig = plt.figure()
    
    for j in range(nb_method):
        g_list_mean = np.mean(g_list[j], axis = 0)
        g_list_min = np.quantile(g_list[j], 0.25, axis = 0)
        g_list_max = np.quantile(g_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(g_list_mean)), g_list_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(g_list_mean)), g_list_min, g_list_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    plt.ylabel(r"$g$", fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence g for inpainting")
    fig.savefig(path_figure+'convergence_g_inpainting.png', dpi = 300)
    plt.show()



    fig = plt.figure()
    
    for j in range(nb_method):
        F_list_mean = np.mean(F_list[j], axis = 0)
        F_list_min = np.quantile(F_list[j], 0.25, axis = 0)
        F_list_max = np.quantile(F_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(F_list_mean)), F_list_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(F_list_mean)), F_list_min, F_list_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    plt.ylabel(r"$F = \lambda f + g$", fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence F for inpainting")
    fig.savefig(path_figure+'convergence_F_inpainting.png', dpi = 300)
    plt.show()

    fig = plt.figure()
    
    for j in range(nb_method):
        nabla_F_list_mean = np.mean(nabla_F_list[j], axis = 0)
        nabla_F_list_min = np.quantile(nabla_F_list[j], 0.25, axis = 0)
        nabla_F_list_max = np.quantile(nabla_F_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(nabla_F_list_mean)), nabla_F_list_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(nabla_F_list_mean)), nabla_F_list_min, nabla_F_list_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    plt.ylabel(r"$\|\nabla F(x^k)\|$", fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence nabla F for inpainting")
    fig.savefig(path_figure+'convergence_nabla_F_inpainting.png', dpi = 300)
    plt.show()

if pars.fig_number == 7:
    path_result = "results/deblurring/CBSD10/"
    method_name = ["RED", r"RiRED $\theta = 0.2$", "Prox-RED", r"Prox-RiRED $\theta = 0.2$"]
    save_name = ["RED", "RiRED", "Prox_RED", "Prox_RiRED"]
    stepsize_method = [0.1, 0.07, 2.0, 5.0]
    nb_method = len(method_name)
    psnr_list = [[] for _ in range(nb_method)]
    residuals = [[] for _ in range(nb_method)]
    i = 9
    k = 0
    path_method = ["GD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/", "GD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/Momentum/theta_0.2/restarting_li/", "PGD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/", "PGD/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/Momentum/theta_0.2/restarting_li/"]
    for j in range(nb_method):
        stepsize = str(stepsize_method[j])
        dic = np.load(path_result + path_method[j] + "/den_level_0.1/lamb_15.0/nb_itr_9/stepsize_"+stepsize+"/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        
        im = dic['restored']
        # print(im.shape())
        # print(np.max(im))
        # print(np.min(im))
        # plt.imsave(path_figure+'restored_9_itr_'+save_name[j]+'.png', im)
        im.save(path_figure+'restored_9_itr_'+save_name[j]+'_psnr_'+str(dic['psnr_restored'])+'.png')



if pars.prep_fig == 7:
    # Preparate the plot of the convergence results of various methods for MRI
    path_result = "results/MRI/MRI_knee/"

    method_name = ["RED", r"RiRED $\theta = 0.2$", "Prox-RED", r"Prox-RiRED $\theta = 0.2$"]
    stepsize_method = [0.7, 0.7, 2.0, 1.0]
    nb_method = len(method_name)
    psnr_list = [[] for _ in range(nb_method)]
    residuals = [[] for _ in range(nb_method)]

    for k in tqdm(range(10)):
        path_method_1 = ["GD", "GD", "PGD", "PGD"]
        path_method_2 = ["", "Momentum/theta_0.2/restarting_li/","", "Momentum/theta_0.2/restarting_li/"]
        for i in range(10):
            for j in range(nb_method):
                stepsize = str(stepsize_method[j])
                dic = np.load(path_result + path_method_1[j] + "/sigma_obs_1.0/denoiser_name_GSDRUNet_grayscale/reduction_factor_4/"+ path_method_2[j] +"den_level_0.01/lamb_1.0/nb_itr_500/stepsize_"+stepsize+"/dict_results_"+str(i)+".npy", allow_pickle=True).item()
                psnr_list[j].append(dic['psnr_list'])
                
                stack_im = dic['stack_images']
                ref = np.sum(np.array(stack_im[0])**2)
                residuals_stack = []
                for l in range(len(stack_im) - 1):
                    residuals_stack.append(np.sum((np.array(stack_im[l+1]) - np.array(stack_im[l]))**2) / ref)
                residuals[j].append(residuals_stack)

    psnr_list = np.array(psnr_list)
    residuals = np.array(residuals)

    dict = {
        'psnr_list' : psnr_list,
        'method_name' : method_name,
        'nb_method' : nb_method,
        'stepsize_method' : stepsize_method,
        'residuals' : residuals,
        }
    np.save(path_figure+"/result_MRI_factor_4", dict)



if pars.fig_number == 7:
    # Generate figure for convergence of various methods for MRI factor 4
    dic = np.load(path_figure+"/result_MRI_factor_4.npy", allow_pickle=True).item()
    
    psnr_list = dic['psnr_list']
    nb_method = dic['nb_method']
    method_name = dic['method_name']
    residuals = dic['residuals']

    fig = plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_indx = [3, 1, 2, 0]

    for j in range(nb_method):
        psnr_mean = np.mean(psnr_list[j], axis = 0)
        # convergence_time = np.sum(psnr_mean < 27.7)
        psnr_std = np.std(psnr_list[j], axis = 0)
        psnr_min = np.quantile(psnr_list[j], 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = method_name[j], color = colors[color_indx[j]])
        # plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    size_number = 17

    # plt.plot([0, 50], [27.7, 27.7], linestyle = "--", color = 'gray')

    plt.xlim(0,499)
    plt.xticks([0, 499], ["0", "500"], fontsize = size_number)
    plt.ylim(28,36)
    plt.yticks([28,36], ["28", "36"], fontsize = size_number)
    # plt.legend()
    fig.savefig(path_figure+'convergence_PSNR_MRI_reduction_factor_4.png', dpi = 300)
    plt.show()


    fig = plt.figure()
    
    for j in range(nb_method):
        residuals_mean = np.mean(np.log10(residuals[j] + 1e-15), axis = 0)
        residuals_min = np.quantile(np.log10(residuals[j] + 1e-15), 0.25, axis = 0)
        residuals_max = np.quantile(np.log10(residuals[j] + 1e-15), 0.75, axis = 0)
        line, = plt.plot(np.arange(len(residuals_mean)), residuals_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(residuals_mean)), residuals_min, residuals_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,499)
    plt.xticks([0, 499], ["0", "500"], fontsize = size_number)

    ticks = np.arange(-15, 1, 1)
    labels = [r"$10^{-15}$" if i == 0 else
            r"$10^0$" if i == len(ticks) - 1 else ""
            for i in range(len(ticks))]
    plt.yticks(ticks, labels, fontsize = size_number)
    plt.ylabel(r"$\|x^{k+1} - x^k\|^2 / \|x^0\|^2$", labelpad=-10, fontsize = size_number)

    # plt.legend()
    fig.savefig(path_figure+'convergence_residuals_MRI_reduction_factor_4.png', dpi = 300)
    plt.show()


if pars.prep_fig == 8:
    # Preparate the plot of the convergence results of various methods for MRI
    path_result = "results/MRI/MRI_knee/"

    method_name = ["RED", r"RiRED $\theta = 0.2$", "Prox-RED", r"Prox-RiRED $\theta = 0.2$"]
    stepsize_method = [0.7, 0.7, 2.0, 1.0]
    nb_method = len(method_name)
    psnr_list = [[] for _ in range(nb_method)]
    residuals = [[] for _ in range(nb_method)]

    for k in tqdm(range(10)):
        path_method_1 = ["GD", "GD", "PGD", "PGD"]
        path_method_2 = ["", "Momentum/theta_0.2/restarting_li/","", "Momentum/theta_0.2/restarting_li/"]
        for i in range(10):
            for j in range(nb_method):
                stepsize = str(stepsize_method[j])
                dic = np.load(path_result + path_method_1[j] + "/sigma_obs_1.0/denoiser_name_GSDRUNet_grayscale/reduction_factor_8/"+ path_method_2[j] +"den_level_0.02/lamb_1.0/nb_itr_500/stepsize_"+stepsize+"/dict_results_"+str(i)+".npy", allow_pickle=True).item()
                psnr_list[j].append(dic['psnr_list'])
                
                stack_im = dic['stack_images']
                ref = np.sum(np.array(stack_im[0])**2)
                residuals_stack = []
                for l in range(len(stack_im) - 1):
                    residuals_stack.append(np.sum((np.array(stack_im[l+1]) - np.array(stack_im[l]))**2) / ref)
                residuals[j].append(residuals_stack)

    psnr_list = np.array(psnr_list)
    residuals = np.array(residuals)

    dict = {
        'psnr_list' : psnr_list,
        'method_name' : method_name,
        'nb_method' : nb_method,
        'stepsize_method' : stepsize_method,
        'residuals' : residuals,
        }
    np.save(path_figure+"/result_MRI_factor_8", dict)

if pars.fig_number == 8:
    # Generate figure for convergence of various methods for inpainting
    dic = np.load(path_figure+"/result_MRI_factor_8.npy", allow_pickle=True).item()
    
    psnr_list = dic['psnr_list']
    nb_method = dic['nb_method']
    method_name = dic['method_name']
    residuals = dic['residuals']

    fig = plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_indx = [3, 1, 2, 0]

    for j in range(nb_method):
        psnr_mean = np.mean(psnr_list[j], axis = 0)
        # convergence_time = np.sum(psnr_mean < 31)
        # print(convergence_time)
        psnr_std = np.std(psnr_list[j], axis = 0)
        psnr_min = np.quantile(psnr_list[j], 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = method_name[j], color = colors[color_indx[j]])
        # plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    size_number = 17

    # plt.plot([0, 50], [31, 31], linestyle = "--", color = 'gray')

    plt.xlim(0,499)
    plt.xticks([0, 499], ["0", "500"], fontsize = size_number)
    plt.ylim(24,32)
    plt.yticks([24,32], ["24", "32"], fontsize = size_number)
    # plt.legend()
    fig.savefig(path_figure+'convergence_PSNR_MRI_reduction_factor_8.png', dpi = 300)
    plt.show()


    fig = plt.figure()
    
    for j in range(nb_method):
        residuals_mean = np.mean(np.log10(residuals[j] + 1e-15), axis = 0)
        residuals_min = np.quantile(np.log10(residuals[j] + 1e-15), 0.25, axis = 0)
        residuals_max = np.quantile(np.log10(residuals[j] + 1e-15), 0.75, axis = 0)
        line, = plt.plot(np.arange(len(residuals_mean)), residuals_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(residuals_mean)), residuals_min, residuals_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,499)
    plt.xticks([0, 499], ["0", "500"], fontsize = size_number)

    ticks = np.arange(-15, 1, 1)
    labels = [r"$10^{-15}$" if i == 0 else
            r"$10^0$" if i == len(ticks) - 1 else ""
            for i in range(len(ticks))]
    plt.yticks(ticks, labels, fontsize = size_number)
    plt.ylabel(r"$\|x^{k+1} - x^k\|^2 / \|x^0\|^2$", labelpad=-10, fontsize = size_number)

    # plt.legend()
    fig.savefig(path_figure+'convergence_residuals_MRI_reduction_factor_8.png', dpi = 300)
    plt.show()



if pars.fig_number == 9:
    path_result = "results/MRI/MRI_knee/"
    method_name = ["RED", r"RiRED $\theta = 0.2$", "Prox-RED", r"Prox-RiRED $\theta = 0.2$"]
    save_name = ["RED", "RiRED", "Prox_RED", "Prox_RiRED"]
    stepsize_method = [0.7, 0.7, 2.0, 1.0]
    nb_method = len(method_name)
    itr_method = [-1, 100, 125, 95]
    i_list = [0,1,2,3]

    path_method_1 = ["GD", "GD", "PGD", "PGD"]
    path_method_2 = ["", "Momentum/theta_0.2/restarting_li/","", "Momentum/theta_0.2/restarting_li/"]
    for i in i_list:
        for j in range(nb_method):
            stepsize = str(stepsize_method[j])
            dic = np.load(path_result + path_method_1[j] + "/sigma_obs_1.0/denoiser_name_GSDRUNet_grayscale/reduction_factor_8/"+ path_method_2[j] +"den_level_0.02/lamb_1.0/nb_itr_500/stepsize_"+stepsize+"/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            
            if j == 0:
                im = dic['clean_image']
                plt.imsave(path_figure+'images_MRI/im_'+str(i)+'_clean_.png', im, cmap = 'gray')
                im = util.tensor2uint(dic['initial_uv'])
                plt.imsave(path_figure+'images_MRI/im_'+str(i)+'_init_.png', im, cmap = 'gray')
                
            im = dic['stack_images'][itr_method[j]]
            im.save(path_figure+'images_MRI/im_'+str(i)+'_restored_'+str(itr_method[j])+'_itr_'+save_name[j]+'_psnr_'+str(dic['psnr_list'][itr_method[j]])+'.png')


if pars.prep_fig == 10:
    # Preparate the plot of the convergence results of various methods for SR
    path_result = "results/SR/CBSD10/"

    method_name = ["RED", r"RiRED $\theta = 0.2$", "Prox-RED", r"Prox-RiRED $\theta = 0.2$"]
    stepsize_method = [0.7, 0.4, 10.0, 10.0]
    nb_method = len(method_name)
    psnr_list = [[] for _ in range(nb_method)]
    residuals = [[] for _ in range(nb_method)]
    for k in tqdm(range(10)):
        path_method_1 = ["GD", "GD", "PGD", "PGD"]
        path_method_2 = ["", "Momentum/theta_0.2/restarting_li/","", "Momentum/theta_0.2/restarting_li/"]
        for i in range(10):
            for j in range(nb_method):
                stepsize = str(stepsize_method[j])
                dic = np.load(path_result + path_method_1[j] + "/sigma_obs_1.0/denoiser_name_GSDRUNet_SoftPlus/kernel_"+str(k)+"/sf_2/"+ path_method_2[j] +"den_level_0.03/lamb_10.0/nb_itr_500/stepsize_"+stepsize+"/dict_results_"+str(i)+".npy", allow_pickle=True).item()
                psnr_list[j].append(dic['psnr_list'])
                
                stack_im = dic['stack_images']
                ref = np.sum(np.array(stack_im[0])**2)
                residuals_stack = []
                for l in range(len(stack_im) - 1):
                    residuals_stack.append(np.sum((np.array(stack_im[l+1]) - np.array(stack_im[l]))**2) / ref)
                residuals[j].append(residuals_stack)

    psnr_list = np.array(psnr_list)
    residuals = np.array(residuals)

    dict = {
        'psnr_list' : psnr_list,
        'method_name' : method_name,
        'nb_method' : nb_method,
        'stepsize_method' : stepsize_method,
        'residuals' : residuals,
        }
    np.save(path_figure+"/result_SR", dict)

if pars.fig_number == 10:
    # Generate figure for convergence of various methods for SR
    dic = np.load(path_figure+"/result_SR.npy", allow_pickle=True).item()
    
    psnr_list = dic['psnr_list']
    nb_method = dic['nb_method']
    method_name = dic['method_name']
    residuals = dic['residuals']

    fig = plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_indx = [3, 1, 2, 0]

    for j in range(nb_method):
        psnr_mean = np.mean(psnr_list[j], axis = 0)
        # convergence_time = np.sum(psnr_mean < 31)
        # print(convergence_time)
        psnr_std = np.std(psnr_list[j], axis = 0)
        psnr_min = np.quantile(psnr_list[j], 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = method_name[j], color = colors[color_indx[j]])
        # plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    size_number = 17

    # plt.plot([0, 50], [31, 31], linestyle = "--", color = 'gray')

    plt.xlim(0,250)
    plt.xticks([0, 250], ["0", "250"], fontsize = size_number)
    # plt.ylim(24,32)
    # plt.yticks([24,32], ["24", "32"], fontsize = size_number)
    # plt.legend()
    fig.savefig(path_figure+'convergence_PSNR_SR.png', dpi = 300)
    plt.show()


    fig = plt.figure()
    
    for j in range(nb_method):
        residuals_mean = np.mean(np.log10(residuals[j] + 1e-15), axis = 0)
        residuals_min = np.quantile(np.log10(residuals[j] + 1e-15), 0.25, axis = 0)
        residuals_max = np.quantile(np.log10(residuals[j] + 1e-15), 0.75, axis = 0)
        line, = plt.plot(np.arange(len(residuals_mean)), residuals_mean, label = method_name[j], color = colors[color_indx[j]])
        plt.fill_between(np.arange(len(residuals_mean)), residuals_min, residuals_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,499)
    plt.xticks([0, 499], ["0", "500"], fontsize = size_number)

    ticks = np.arange(-15, 1, 1)
    labels = [r"$10^{-15}$" if i == 0 else
            r"$10^0$" if i == len(ticks) - 1 else ""
            for i in range(len(ticks))]
    plt.yticks(ticks, labels, fontsize = size_number)
    plt.ylabel(r"$\|x^{k+1} - x^k\|^2 / \|x^0\|^2$", labelpad=-10, fontsize = size_number)

    # plt.legend()
    fig.savefig(path_figure+'convergence_residuals_SR.png', dpi = 300)
    plt.show()



# if pars.fig_number == 2:
#     #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.2
#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1/B_"

#     B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
#     fig = plt.figure()
    
#     for B in B_list:
#         psnr_list = []
#         for i in range(5):
#             dic = np.load(path_result + str(B) + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#             psnr_list.append(dic['psnr_list'])
#         psnr_list = np.array(psnr_list)
#         psnr_mean = np.mean(psnr_list, axis = 0)
#         plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
#     path_result = "results/set5/RED_level_0.1_lamb18"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    

#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.1$", linestyle='--')

#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.2$", linestyle='--')

#     plt.legend()
#     plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.1$ on $5$ images" +"\n from CBSD68 dataset with various restarting parameter B")
#     fig.savefig(path_figure+'/result_momentum_various_B_theta_0.1.png', dpi = 300)
#     plt.show()


# if pars.fig_number == 3:
#     #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.15
#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.05/B_"

#     B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
#     fig = plt.figure()
    
#     for B in B_list:
#         psnr_list = []
#         for i in range(5):
#             dic = np.load(path_result + str(B) + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#             psnr_list.append(dic['psnr_list'])
#         psnr_list = np.array(psnr_list)
#         psnr_mean = np.mean(psnr_list, axis = 0)
#         plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
#     path_result = "results/set5/RED_level_0.1_lamb18"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    

#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.05"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.05$", linestyle='--')

#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.2$", linestyle='--')

#     plt.legend()
#     plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.1$ on $5$ images" +"\n from CBSD68 dataset with various restarting parameter B")
#     fig.savefig(path_figure+'/result_momentum_various_B_theta_0.05.png', dpi = 300)
#     plt.show()



# if pars.fig_number == 4:
#     #generate figure for deblurring on 5 images with various restarting Su
#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_"

#     theta_list = [0.01,0.05,0.1,0.2]
#     fig = plt.figure()
    
#     for theta in theta_list:
#         psnr_list = []
#         for i in range(5):
#             dic = np.load(path_result + str(theta) + "restarting_su/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#             psnr_list.append(dic['psnr_list'])
#         psnr_list = np.array(psnr_list)
#         psnr_mean = np.mean(psnr_list, axis = 0)
#         plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"Su $\theta = $"+str(theta))
    
#     path_result = "results/set5/RED_level_0.1_lamb18"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')


#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart '+r"$\theta = 0.2$", linestyle='--')

#     plt.legend()
#     plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.1$ on $5$ images" +"\n from CBSD68 dataset with Su restarting")
#     fig.savefig(path_figure+'/result_momentum_restarting_su.png', dpi = 300)
#     plt.show()


# if pars.fig_number == 5:
#     #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.2 for 200 iterations
#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1/B_"

#     B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
#     fig = plt.figure()
    
#     for B in B_list:
#         psnr_list = []
#         for i in range(5):
#             dic = np.load(path_result + str(B) + "/nb_itr_200/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#             psnr_list.append(dic['psnr_list'])
#         psnr_list = np.array(psnr_list)
#         psnr_mean = np.mean(psnr_list, axis = 0)
#         plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
#     path_result = "results/set5/RED_level_0.1_lamb18/nb_itr_200"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    

#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1/nb_itr_200"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.1$", linestyle='--')

#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.2$", linestyle='--')

#     plt.legend()
#     plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.1$ on $5$ images" +"\n from CBSD68 dataset with various restarting parameter B")
#     fig.savefig(path_figure+'/result_momentum_various_B_theta_0.1_nb_itr_200.png', dpi = 300)
#     plt.show()


# if pars.fig_number == 6:
#     #generate figure for deblurring on 10 images with various momentum parameter without restart
#     path_result = "results/CBSD10/GD_den_level_0.1_lamb_15.0/Momentum/theta_"

#     theta_list = [0.01, 0.1, 0.2, 0.9]
#     fig = plt.figure()
    
#     for theta in theta_list:
#         psnr_list = []
#         for i in range(10):
#             dic = np.load(path_result + str(theta) + "/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#             psnr_list.append(dic['psnr_list'])
#         psnr_list = np.array(psnr_list)
#         psnr_mean = np.mean(psnr_list, axis = 0)
#         psnr_std = np.std(psnr_list, axis = 0)
#         psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
#         psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
#         line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$\theta = $"+str(theta))
#         plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    
#     path_result = "results/CBSD10/GD_den_level_0.1_lamb_18/stepsize_0.1"
#     psnr_list = []
#     for i in range(10):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     psnr_std = np.std(psnr_list, axis = 0)
#     psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
#     psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
#     line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
#     plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

#     path_result = "results/CBSD10/PGD_den_level_0.1_lamb_18/stepsize_2.0"
#     psnr_list = []
#     for i in range(10):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     psnr_std = np.std(psnr_list, axis = 0)
#     psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
#     psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
#     line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'PGD')
#     plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

#     path_result = "results/CBSD10/PGD_den_level_0.1_lamb_15.0/Momentum/theta_0.2/stepsize_5.0"
#     psnr_list = []
#     for i in range(10):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     psnr_std = np.std(psnr_list, axis = 0)
#     psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
#     psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
#     line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r'PGD $\theta = 0.2$')
#     plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())


#     plt.xlim(0,49)
#     plt.xticks([0,49], ["0","50"])
#     plt.ylim(15,30)
#     plt.yticks([15,30], ["15","30"])
#     plt.legend()
#     plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ on $10$ images" +"\n" + r"from CBSD68 dataset with various momentum parameter $\theta$")
#     fig.savefig(path_figure+'/result_momentum_various_theta_without_restart.png', dpi = 300)
#     plt.show()

if pars.table_number == 0:
    #generate the result of inpainting for parameters tunning for PGD without momentum

    path_result = "results/inpainting/set5/p_0.2/"

    for sigma in [0.08]:
        for lamb in [1., 3., 5., 10., 15.]:
            n = 5
            output_psnr = []
            output_ssim = []
            for i in range(n):
                dic_Diff = np.load(path_result + "PGD/sigma_obs_1.0/den_level_{}/lamb_{}/nb_itr_500/stepsize_5.0/dict_results_{}.npy".format(sigma, lamb, i), allow_pickle=True).item()
                output_psnr.append(dic_Diff["psnr_restored"])
                output_ssim.append(dic_Diff["ssim_restored"])

            output_psnr = np.array(output_psnr)
            output_ssim = np.array(output_ssim)
            print("Sigma = {}, Lambda = {}".format(sigma, lamb), "PSNR/SSIM : {:.2f} & {:.2f}".format(np.mean(output_psnr),np.mean(output_ssim)))

if pars.table_number == 1:
    #generate the result of deblurring for parameters tunning for PGD/GD without momentum

    path_result = "results/deblurring/set5/"

    for sigma in [0.09, 0.1, 0.11]:
        for lamb in [13., 15., 18.]:
            n = 5
            output_psnr = []
            output_ssim = []
            for i in range(n):
                dic_Diff = np.load(path_result + "GD/Momentum/theta_0.2/den_level_{}/lamb_{}/stepsize_0.07/dict_results_{}.npy".format(sigma, lamb, i), allow_pickle=True).item()
                output_psnr.append(dic_Diff["psnr_restored"])
                output_ssim.append(dic_Diff["ssim_restored"])

            output_psnr = np.array(output_psnr)
            output_ssim = np.array(output_ssim)
            print("Sigma = {}, Lambda = {}".format(sigma, lamb), "PSNR/SSIM : {:.2f} & {:.2f}".format(np.mean(output_psnr),np.mean(output_ssim)))


if pars.table_number == 2:
    #generate the result of deblurring for various method with 10 various kernel of blur

    path_result = "results/deblurring/CBSD10/"
    method_name = ["GD", r"GD $\theta = 0.2$", "PGD", r"PGD $\theta = 0.2$"]
    stepsize_method = [0.1, 0.07, 2.0, 5.0]
    nb_method = len(method_name)
    output_psnr = [[] for _ in range(nb_method)]
    output_ssim = [[] for _ in range(nb_method)]

    for k in tqdm(range(10)):
        path_method = ["GD/kernel_"+str(k)+"/", "GD/kernel_"+str(k)+"/Momentum/theta_0.2/restarting_li/", "PGD/kernel_"+str(k)+"/", "PGD/kernel_"+str(k)+"/Momentum/theta_0.2/restarting_li/"]
        for i in range(10):
            for j in range(nb_method):
                stepsize = stepsize_method[j]
                dic = np.load(path_result + path_method[j] + "den_level_0.1/lamb_15.0/stepsize_"+str(stepsize)+"/dict_results_{}.npy".format(i), allow_pickle=True).item()
                output_psnr[j].append(dic["psnr_restored"])
                output_ssim[j].append(dic["ssim_restored"])

    output_psnr = np.array(output_psnr)
    output_ssim = np.array(output_ssim)
    for j in range(nb_method):
        print(method_name[j], " : PSNR/SSIM : {:.2f} & {:.2f}".format(np.mean(output_psnr[j]),np.mean(output_ssim[j])))



if pars.table_number == 3:
    #generate the result of the grid-search for speckle L = 10
    path_result = "results/SR/set5/GD/sigma_obs_1.0/denoiser_name_GSDRUNet_SoftPlus/"
    lamb_list = [5.0, 10.0, 15.0]
    sigma_list = [0.01]
    step_list = [0.7,0.7,0.5]
    kernel_list = [0, 1, 2, 8, 9]
    n = 4
    for std in sigma_list:
        for i, lamb in enumerate(lamb_list):
            output_psnr = []
            for j in range(n):
                for k in kernel_list:
                    dic_RED = np.load(path_result + "kernel_"+ str(k) +"/sf_2/den_level_" + str(std) +"/lamb_" + str(lamb) +"/nb_itr_500/stepsize_"+str(step_list[i])+"/dict_results_"+str(j)+".npy", allow_pickle=True).item()
                    output_psnr.append(dic_RED["psnr_restored"])
            print("Sigma ",std,"Lambda ",lamb, ":", np.mean(output_psnr))


# if pars.fig_number == 4:
#     #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.01
#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.01/B_"

#     B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
#     B_list = [1000.0]
#     fig = plt.figure()

#     for B in B_list:
#         psnr_list = []
#         for i in range(5):
#             dic = np.load(path_result + str(B) + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#             psnr_list.append(dic['psnr_list'])
#         psnr_list = np.array(psnr_list)
#         psnr_mean = np.mean(psnr_list, axis = 0)
#         plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = "theta = 0.01")
#         path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1/B_"

#     B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
#     B_list = [1000.0]


#     for B in B_list:
#         psnr_list = []
#         for i in range(5):
#             dic = np.load(path_result + str(B) + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#             psnr_list.append(dic['psnr_list'])
#         psnr_list = np.array(psnr_list)
#         psnr_mean = np.mean(psnr_list, axis = 0)
#         plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = "theta = 0.1")
#     path_result = "results/set5/RED_level_0.1_lamb18"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')


#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.01"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.01$", linestyle='--')
#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.1$", linestyle='--')


#     plt.legend()
#     plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $B = 1000$ on $5$ images" +"\n from CBSD68 dataset with different momentum parameter theta")
#     fig.savefig(path_figure+'result_momentum_various_B_theta_0.05.png', dpi = 300)
#     plt.show()