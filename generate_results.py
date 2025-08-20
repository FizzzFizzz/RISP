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

    path_result = "results/deblurring/CBSD10/"
    method_name = ["GD", r"GD $\theta = 0.2$", "PGD", r"PGD $\theta = 0.2$"]
    stepsize_method = [0.1, 0.07, 2.0, 5.0]
    nb_method = len(method_name)
    psnr_list = [[] for _ in range(nb_method)]
    residuals = [[] for _ in range(nb_method)]

    for k in tqdm(range(10)):
        path_method = ["GD/kernel_"+str(k)+"/", "GD/kernel_"+str(k)+"/Momentum/theta_0.2/restarting_li/", "PGD/kernel_"+str(k)+"/", "PGD/kernel_"+str(k)+"/Momentum/theta_0.2/restarting_li/"]
        for i in range(10):
            for j in range(nb_method):
                stepsize = str(stepsize_method[j])
                dic = np.load(path_result + path_method[j] + "/den_level_0.1/lamb_15.0/stepsize_"+stepsize+"/dict_results_"+str(i)+".npy", allow_pickle=True).item()
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
    np.save(path_figure+"/result_deblurring_expe", dict)


if pars.fig_number == 2:
    # Generate figure for convergence of the method for delurring
    dic = np.load(path_figure+"/result_deblurring_expe.npy", allow_pickle=True).item()
    
    psnr_list = dic['psnr_list']
    nb_method = dic['nb_method']
    method_name = dic['method_name']
    residuals = dic['residuals']

    fig = plt.figure()

    for j in range(nb_method):
        psnr_mean = np.mean(psnr_list[j], axis = 0)
        convergence_time = np.sum(psnr_mean < 27.7)
        psnr_std = np.std(psnr_list[j], axis = 0)
        psnr_min = np.quantile(psnr_list[j], 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list[j], 0.75, axis = 0)
        line, = plt.plot(np.arange(convergence_time), psnr_mean[:convergence_time], label = method_name[j])
        # plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    size_number = 15

    plt.plot([0, 50], [27.7, 27.7], linestyle = "--", color = 'gray')

    plt.xlim(0,49)
    plt.xticks([0, 9, 20, 38, 49], ["0", "9", "20", "38", "50"], fontsize = size_number)
    plt.ylim(18,28)
    plt.yticks([18, 27.7], ["18", "27.7"], fontsize = size_number)
    plt.legend()
    plt.title(r"Convergence PSNR for deblurring")
    fig.savefig(path_figure+'convergence_PSNR_deblurring.png', dpi = 300)
    plt.show()


    fig = plt.figure()
    
    for j in range(nb_method):
        residuals_mean = np.mean(np.log10(residuals[j]), axis = 0)
        residuals_min = np.quantile(np.log10(residuals[j]), 0.25, axis = 0)
        residuals_max = np.quantile(np.log10(residuals[j]), 0.75, axis = 0)
        line, = plt.plot(np.arange(len(residuals_mean)), residuals_mean, label = method_name[j])
        plt.fill_between(np.arange(len(residuals_mean)), residuals_min, residuals_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0, 49], ["0", "50"], fontsize = size_number)

    ticks = np.arange(-4, 1, 1)
    labels = [r"$10^{-4}$" if i == 0 else
            r"$10^0$" if i == len(ticks) - 1 else ""
            for i in range(len(ticks))]
    plt.yticks(ticks, labels, fontsize = size_number)
    plt.ylabel(r"$\|x^{k+1} - x^k\|^2 / \|x^0\|^2$", labelpad=-10, fontsize = size_number)

    plt.legend()
    plt.title(r"Convergence residuals for deblurring")
    fig.savefig(path_figure+'convergence_residuals_deblurring.png', dpi = 300)
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
    fig.savefig(path_figure+'/result_deblurring_various_theta_with_restart.png', dpi = 300)


# if pars.fig_number == 1:
#     #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.2
#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2/B_"

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
    

#     path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
#     psnr_list = []
#     for i in range(5):
#         dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
#         psnr_list.append(dic['psnr_list'])
#     psnr_list = np.array(psnr_list)
#     psnr_mean = np.mean(psnr_list, axis = 0)
#     plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart', linestyle='--')

#     plt.legend()
#     plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.2$ on $5$ images" +"\n from CBSD68 dataset with various restarting parameter B")
#     fig.savefig(path_figure+'/result_momentum_various_B_theta_0.2.png', dpi = 300)
#     plt.show()



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