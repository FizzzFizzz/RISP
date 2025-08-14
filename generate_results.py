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
    #generate figure for deblurring on 10 images with various momentum parameter
    path_result = "results/CBSD10/GD_den_level_0.1_lamb_15.0/Momentum/theta_"

    theta_list = [0.01, 0.1, 0.2, 0.9]
    fig = plt.figure()
    
    for theta in theta_list:
        psnr_list = []
        for i in range(10):
            dic = np.load(path_result + str(theta) + "/restarting_li/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        psnr_std = np.std(psnr_list, axis = 0)
        psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$\theta = $"+str(theta))
        plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    
    path_result = "results/CBSD10/GD_den_level_0.1_lamb_18/stepsize_0.1"
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

    path_result = "results/CBSD10/PGD_den_level_0.1_lamb_18/stepsize_2.0"
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    psnr_std = np.std(psnr_list, axis = 0)
    psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
    psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
    line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'PGD')
    plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    path_result = "results/CBSD10/PGD_den_level_0.1_lamb_15.0/Momentum/theta_0.2/restarting_li/stepsize_5.0"
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    psnr_std = np.std(psnr_list, axis = 0)
    psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
    psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
    line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r'PGD $\theta = 0.2$')
    plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    plt.xlim(0,49)
    plt.xticks([0,49], ["0","50"])
    plt.ylim(15,30)
    plt.yticks([15,30], ["15","30"])
    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ on $10$ images" +"\n" + r"from CBSD68 dataset with various momentum parameter $\theta$")
    fig.savefig(path_figure+'/result_momentum_various_theta.png', dpi = 300)
    plt.show()


if pars.fig_number == 1:
    #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.2
    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2/B_"

    B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
    fig = plt.figure()
    
    for B in B_list:
        psnr_list = []
        for i in range(5):
            dic = np.load(path_result + str(B) + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
    path_result = "results/set5/RED_level_0.1_lamb18"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    

    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart', linestyle='--')

    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.2$ on $5$ images" +"\n from CBSD68 dataset with various restarting parameter B")
    fig.savefig(path_figure+'/result_momentum_various_B_theta_0.2.png', dpi = 300)
    plt.show()



if pars.fig_number == 2:
    #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.2
    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1/B_"

    B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
    fig = plt.figure()
    
    for B in B_list:
        psnr_list = []
        for i in range(5):
            dic = np.load(path_result + str(B) + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
    path_result = "results/set5/RED_level_0.1_lamb18"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    

    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.1$", linestyle='--')

    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.2$", linestyle='--')

    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.1$ on $5$ images" +"\n from CBSD68 dataset with various restarting parameter B")
    fig.savefig(path_figure+'/result_momentum_various_B_theta_0.1.png', dpi = 300)
    plt.show()


if pars.fig_number == 3:
    #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.15
    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.05/B_"

    B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
    fig = plt.figure()
    
    for B in B_list:
        psnr_list = []
        for i in range(5):
            dic = np.load(path_result + str(B) + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
    path_result = "results/set5/RED_level_0.1_lamb18"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    

    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.05"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.05$", linestyle='--')

    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.2$", linestyle='--')

    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.1$ on $5$ images" +"\n from CBSD68 dataset with various restarting parameter B")
    fig.savefig(path_figure+'/result_momentum_various_B_theta_0.05.png', dpi = 300)
    plt.show()



if pars.fig_number == 4:
    #generate figure for deblurring on 5 images with various restarting Su
    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_"

    theta_list = [0.01,0.05,0.1,0.2]
    fig = plt.figure()
    
    for theta in theta_list:
        psnr_list = []
        for i in range(5):
            dic = np.load(path_result + str(theta) + "restarting_su/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"Su $\theta = $"+str(theta))
    
    path_result = "results/set5/RED_level_0.1_lamb18"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')


    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart '+r"$\theta = 0.2$", linestyle='--')

    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.1$ on $5$ images" +"\n from CBSD68 dataset with Su restarting")
    fig.savefig(path_figure+'/result_momentum_restarting_su.png', dpi = 300)
    plt.show()


if pars.fig_number == 5:
    #generate figure for deblurring on 5 images with various restarting parameter and the momentum parameter theta = 0.2 for 200 iterations
    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1/B_"

    B_list = [100.0, 1000.0, 3000.0, 5000.0, 7000.0, 10000.0, 100000.0]
    fig = plt.figure()
    
    for B in B_list:
        psnr_list = []
        for i in range(5):
            dic = np.load(path_result + str(B) + "/nb_itr_200/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$B = $"+str(B)[:-2])
    
    path_result = "results/set5/RED_level_0.1_lamb18/nb_itr_200"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'RED')
    

    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.1/nb_itr_200"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.1$", linestyle='--')

    path_result = "results/set5/RED_level_0.1_lamb18_Momentum_theta_0.2"
    psnr_list = []
    for i in range(5):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'No restart'+r"$\theta = 0.2$", linestyle='--')

    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ and $\theta = 0.1$ on $5$ images" +"\n from CBSD68 dataset with various restarting parameter B")
    fig.savefig(path_figure+'/result_momentum_various_B_theta_0.1_nb_itr_200.png', dpi = 300)
    plt.show()


if pars.fig_number == 6:
    #generate figure for deblurring on 10 images with various momentum parameter without restart
    path_result = "results/CBSD10/GD_den_level_0.1_lamb_15.0/Momentum/theta_"

    theta_list = [0.01, 0.1, 0.2, 0.9]
    fig = plt.figure()
    
    for theta in theta_list:
        psnr_list = []
        for i in range(10):
            dic = np.load(path_result + str(theta) + "/stepsize_0.07/dict_results_"+str(i)+".npy", allow_pickle=True).item()
            psnr_list.append(dic['psnr_list'])
        psnr_list = np.array(psnr_list)
        psnr_mean = np.mean(psnr_list, axis = 0)
        psnr_std = np.std(psnr_list, axis = 0)
        psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
        psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
        line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r"$\theta = $"+str(theta))
        plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())
    
    path_result = "results/CBSD10/GD_den_level_0.1_lamb_18/stepsize_0.1"
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

    path_result = "results/CBSD10/PGD_den_level_0.1_lamb_18/stepsize_2.0"
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    psnr_std = np.std(psnr_list, axis = 0)
    psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
    psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
    line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = 'PGD')
    plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())

    path_result = "results/CBSD10/PGD_den_level_0.1_lamb_15.0/Momentum/theta_0.2/stepsize_5.0"
    psnr_list = []
    for i in range(10):
        dic = np.load(path_result + "/dict_results_"+str(i)+".npy", allow_pickle=True).item()
        psnr_list.append(dic['psnr_list'])
    psnr_list = np.array(psnr_list)
    psnr_mean = np.mean(psnr_list, axis = 0)
    psnr_std = np.std(psnr_list, axis = 0)
    psnr_min = np.quantile(psnr_list, 0.25, axis = 0)
    psnr_max = np.quantile(psnr_list, 0.75, axis = 0)
    line, = plt.plot(np.arange(len(psnr_mean)), psnr_mean, label = r'PGD $\theta = 0.2$')
    plt.fill_between(np.arange(len(psnr_mean)), psnr_min, psnr_max, alpha=0.1, color=line.get_color())


    plt.xlim(0,49)
    plt.xticks([0,49], ["0","50"])
    plt.ylim(15,30)
    plt.yticks([15,30], ["15","30"])
    plt.legend()
    plt.title(r"PSNR evolution on deblurring with $\sigma = 12.5/255$ on $10$ images" +"\n" + r"from CBSD68 dataset with various momentum parameter $\theta$")
    fig.savefig(path_figure+'/result_momentum_various_theta_without_restart.png', dpi = 300)
    plt.show()



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