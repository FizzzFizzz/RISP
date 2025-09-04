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
from argparse import ArgumentParser
from algorithm import *
from forward_model import *
from time import time

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, default="models_ckpt/drunet_color.pth", help = "The path for the DRUNet pretrained weights")
parser.add_argument('--denoiser_name', type=str, default="GSDRUNet", help = "Type of denoiser, DRUNet, GSDRUNet or GSDRUNet_SoftPlus are implemented")
parser.add_argument('--Pb', type=str, default="deblurring", help = "Inverse problem to tackle: deblurring, inpainting, ODT")
parser.add_argument('--gpu_number', type=int, default=0, help = "the GPU number")
parser.add_argument('--Nesterov', dest='Nesterov', action='store_true')
parser.set_defaults(Nesterov=False)
parser.add_argument('--momentum', dest='momentum', action='store_true')
parser.set_defaults(momentum=False)
parser.add_argument('--grayscale', dest='grayscale', action='store_true')
parser.set_defaults(grayscale=False)
parser.add_argument('--restarting_su', dest='restarting_su', action='store_true')
parser.set_defaults(restarting_su=False)
parser.add_argument('--restarting_li', dest='restarting_li', action='store_true')
parser.set_defaults(restarting_li=False)
parser.add_argument('--alg', type=str, default="GD", help = "type of step on the data-fidelity, if alg = 'GD' it is a graident step on the data-fidelity, if alg = 'PGD' it is a proximal step on the data-fidelity")
parser.add_argument('--r', type=int, default=3, help = "Parameter for the Generalized Nesterov momentum")
parser.add_argument('--B', type=float, default=5000., help = "Parameter restarting criterion proposed by Li")
parser.add_argument('--lamb', type=float, default=18, help = "Regularization parameter")
parser.add_argument('--denoiser_level', type=float, default=0.1, help = "Denoiser level in [0.,1.]")
parser.add_argument('--sigma_obs', type=float, default=12.75, help = "Standard variation of the noise in the observation in [0.,255.]")
parser.add_argument('--dataset_name', type=str, default='set1', help = "Name of the dataset of image to restore")
parser.add_argument('--kernel_name', type=str, default='levin_6.png', help = "Name of the kernel of blur")
parser.add_argument('--kernel_index', type=int, default=5, help = "Index of the kernel of blur")
parser.add_argument('--stepsize', type=float, default=0.02, help = "Stepsize of the gradient descent algorithm")
parser.add_argument('--nb_itr', type=int, default=50, help = "Number of iterations of the algorithm")
parser.add_argument('--theta', type=float, default=0.9, help = "Momentum parameter")
parser.add_argument('--start_im_indx', type=int, default=0, help = "Rank of the image to start from in the dataset")
parser.add_argument('--p', type=float, default=0.5, help = "Proportion of viewed pixels for inpainting with random mask")
parser.add_argument('--L', type=int, default=10, help = "Number of looks for image depseckling")
parser.add_argument('--reduction_factor', type=int, default=8, help = "Factor of acceleration for MRI")
parser.add_argument('--dont_save_images', dest='dont_save_images', action='store_true')
parser.set_defaults(dont_save_images=False)
parser.add_argument('--save_each_itr', dest='save_each_itr', action='store_true')
parser.set_defaults(save_each_itr=False)
hparams = parser.parse_args()

Pb = hparams.Pb
denoiser_name = hparams.denoiser_name
model_path = hparams.model_path
nb_itr = hparams.nb_itr
# n_channels = hparams.n_channels
device = torch.device('cuda:'+str(hparams.gpu_number) if torch.cuda.is_available() else 'cpu')

# Parameters setting
denoiser_level = hparams.denoiser_level
lamb = hparams.lamb
sigma_obs = hparams.sigma_obs
r = hparams.r
B = hparams.B
theta = hparams.theta
Nesterov = hparams.Nesterov
momentum = hparams.momentum
grayscale = hparams.grayscale
if Pb == 'MRI' and not('--grayscale' in sys.argv):
    grayscale = True
restarting_su = hparams.restarting_su
restarting_li = hparams.restarting_li
stepsize = hparams.stepsize
dont_save_images = hparams.dont_save_images
save_each_itr = hparams.save_each_itr
alg = hparams.alg
Average_PSNR = 0
L = hparams.L

# Set input image paths
input_path = os.path.join('datasets', hparams.dataset_name)
input_paths = os_sorted([os.path.join(input_path, im_name) for im_name in os.listdir(input_path)])

for i in range(hparams.start_im_indx, len(input_paths)):
    clean_image_path = input_paths[i]
    model = PnP(nb_itr = nb_itr, denoiser_name = denoiser_name, device = device, Pb = Pb, sigma_obs = sigma_obs)
    model.eval()
    model.net.eval()
    model.to(device)
    model.net.to(device)

    # Load the clean image
    if not(grayscale):
        clean_image = util.imread_uint(clean_image_path, 3)
        clean_image = util.single2tensor3(clean_image).unsqueeze(0) /255.
    else:
        clean_image = util.imread_uint(clean_image_path, 1)
        clean_image = util.single2tensor3(clean_image).unsqueeze(0) /255.

    if Pb == "deblurring":
        if '--kernel_index' in sys.argv:
            k_index = hparams.kernel_index
            kernel_path = os.path.join('utils/kernels', 'Levin09.mat')
            kernels = hdf5storage.loadmat(kernel_path)['kernels']
            # Kernels follow the order given in the paper (Table 2). The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
            if k_index == 8: # Uniform blur
                kernel = (1/81)*np.ones((9,9))
            elif k_index == 9:  # Gaussian blur
                kernel = deblur.matlab_style_gauss2D(shape=(25,25),sigma=1.6)
            else : # Motion blur
                kernel = kernels[0, k_index]
            kernel = torch.from_numpy(np.ascontiguousarray(kernel)).float().unsqueeze(0).unsqueeze(0)
        else:
            kernel_path = 'utils/kernels/'+hparams.kernel_name
            kernel = util.imread_uint(kernel_path,1)
            kernel = util.single2tensor3(kernel).unsqueeze(0) / 255.
            kernel = kernel / torch.sum(kernel)
        model.kernel = kernel
        observation = gen_data(model, clean_image, sigma_obs)
        model.kernel = model.kernel.to(device)
        clean_image = clean_image.to(device)
        initial_uv = observation.clone()

    elif Pb == "inpainting":
        model.p = hparams.p
        clean_image = clean_image.to(device)
        observation, mask = gen_data(model, clean_image, sigma_obs)
        model.mask = mask
        initial_uv = mask*observation.clone() + 0.5 * (1 - mask)

    elif Pb == "MRI":
        clean_image = clean_image.to(device)
        model.numLines = int(clean_image.shape[-1] / hparams.reduction_factor)
        observation, mask, pseudo_inverse = gen_data(model, clean_image, sigma_obs)
        model.M = mask
        initial_uv = pseudo_inverse

    elif Pb == "speckle":
        clean_image = clean_image.to(device)
        model.noise_model = "speckle"
        model.L = L
        observation = gen_data(model, clean_image)
        initial_uv = observation

    time_restore = time()

    # Run RED algorithm with or without momentum
    with torch.no_grad():
        model.forward(initial_uv, observation, clean_image, sigma_obs, lamb, denoiser_level, theta, r, B, Nesterov, momentum, restarting_su, restarting_li, stepsize, alg)

    time_restore = time() - time_restore
    print("The time of restoration is : ", time_restore)

    psnr_list = model.res['psnr']
    ssim_list = model.res['ssim']
    print("Restored image PSNR = {:.2f}".format(psnr_list[-1]))
    Average_PSNR += psnr_list[-1]
    if restarting_su or restarting_li:
        print("Number of restarting activation = {}".format(model.nb_restart_activ))
    
    # Define the path for saving the experiment
    savepth = 'results/'+Pb+'/'+hparams.dataset_name
    if '--p' in sys.argv:
        savepth = os.path.join(savepth, 'p_'+str(model.p))
        os.makedirs(savepth, exist_ok = True)
    savepth = savepth + "/" + alg + "/"
    os.makedirs(savepth, exist_ok = True)
    if '--sigma_obs' in sys.argv:
        savepth = os.path.join(savepth, 'sigma_obs_'+str(sigma_obs))
        os.makedirs(savepth, exist_ok = True)
    if '--denoiser_name' in sys.argv:
        savepth = os.path.join(savepth, 'denoiser_name_'+denoiser_name)
        os.makedirs(savepth, exist_ok = True)
    if '--kernel_index' in sys.argv:
        savepth = os.path.join(savepth, 'kernel_'+str(k_index))
        os.makedirs(savepth, exist_ok = True)
    if '--L' in sys.argv:
        savepth = os.path.join(savepth, 'L_'+str(L))
        os.makedirs(savepth, exist_ok = True)
    if '--reduction_factor' in sys.argv:
        savepth = os.path.join(savepth, 'reduction_factor_'+str(hparams.reduction_factor))
        os.makedirs(savepth, exist_ok = True)
    if Nesterov:
        savepth = savepth + "/Nesterov"
        savepth = os.path.join(savepth, 'r_'+str(r))
        os.makedirs(savepth, exist_ok = True)
    elif momentum:
        savepth = savepth + "/Momentum"
        savepth = os.path.join(savepth, 'theta_'+str(theta))
        os.makedirs(savepth, exist_ok = True)
    if restarting_su:
        savepth = os.path.join(savepth, "restarting_su")
        os.makedirs(savepth, exist_ok = True)
    if restarting_li:
        savepth = os.path.join(savepth, "restarting_li")
        os.makedirs(savepth, exist_ok = True)
    savepth = os.path.join(savepth, "den_level_{}".format(denoiser_level))
    os.makedirs(savepth, exist_ok = True)
    savepth = os.path.join(savepth, "lamb_{}".format(lamb))
    os.makedirs(savepth, exist_ok = True)
    if '--B' in sys.argv:
        savepth = os.path.join(savepth, 'B_'+str(hparams.B))
        os.makedirs(savepth, exist_ok = True)
    if '--nb_itr' in sys.argv:
        savepth = os.path.join(savepth, 'nb_itr_'+str(hparams.nb_itr))
        os.makedirs(savepth, exist_ok = True)
    if '--stepsize' in sys.argv:
        savepth = os.path.join(savepth, 'stepsize_'+str(hparams.stepsize))
        os.makedirs(savepth, exist_ok = True)

    print("Results are saved in the path : ", savepth)

    if not(dont_save_images):
        if grayscale :
            colormap = 'gray'
        else:
            colormap = 'viridis'
        if save_each_itr:
            savepth_img = savepth+"/set_img_{}/".format(i)
            os.makedirs(savepth_img, exist_ok = True)
            for j in range(len(model.res['image'])):
                model.res['image'][j].save(savepth_img + 'iterations_{}.png'.format(j))
        
        model.res['image'][-1].save(savepth + "/{}_restored_img.png".format(i))
        clean_img_uint = util.tensor2uint(clean_image)
        obs_uint = util.tensor2uint(observation)
        plt.imsave(savepth + "/{}_clean_img.png".format(i), clean_img_uint, cmap = colormap)
        plt.imsave(savepth + "/{}_observation.png".format(i), obs_uint, cmap = colormap)

        itr_list = range(len(psnr_list))
        plt.clf()
        plt.plot(itr_list, psnr_list, '-', alpha=0.8, linewidth=1.5)
        plt.xlabel('iter')
        plt.ylabel('PSNR')
        plt.savefig(savepth+"/{}_PSNR_list.png".format(i))
        plt.clf()
        plt.plot(itr_list, ssim_list, '-', alpha=0.8, linewidth=1.5)
        plt.xlabel('iter')
        plt.ylabel('SSIM')
        plt.savefig(savepth+"/{}_SSIM_list.png".format(i))
    
    dict = {
            'clean_image' : util.tensor2uint(clean_image),
            'observation' : util.tensor2uint(observation),
            'initial_uv' : initial_uv,
            'sigma_obs' : sigma_obs,
            'lamb' : lamb,
            'denoiser_level' : denoiser_level,
            'r' : r,
            'Nesterov' : Nesterov,
            'restarting_su' : restarting_su,
            'stack_images' : model.res['image'],
            'clean_image_path' : clean_image_path,
            'psnr_list' : psnr_list,
            'ssim_list' : ssim_list,
            'psnr_restored' : psnr_list[-1],
            'ssim_restored' : ssim_list[-1],
            'restored' : model.res['image'][-1],
            'nb_restart_activ' : model.nb_restart_activ,
            'grayscale' : grayscale,
        }
    
    if Pb == "deblurring":
        dict['kernel'] = util.tensor2uint(kernel)
        dict['kernel_path'] = kernel_path
    
    if Pb == "inpainting":
        dict['p'] = model.p
        dict['mask'] = mask
    
    if Pb == "MRI":
        dict['reduction_factor'] = hparams.reduction_factor
        dict['numLines'] = model.numLines
    
    np.save(savepth+"/dict_results_{}".format(i), dict)

print("Average restored PSNR on the dataset = {:.2f}".format(Average_PSNR / len(input_paths)))
