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

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, default="models_ckpt/drunet_color.pth", help = "The path for the DRUNet pretrained weights")
parser.add_argument('--Pb', type=str, default="deblurring", help = "Inverse problem to tackle")
parser.add_argument('--n_channels', type=int, default=3, help = "number of channels of the image, by default RGB")
parser.add_argument('--gpu_number', type=int, default=0, help = "the GPU number")
parser.add_argument('--Nesterov', dest='Nesterov', action='store_true')
parser.set_defaults(Nesterov=False)
parser.add_argument('--momentum', dest='momentum', action='store_true')
parser.set_defaults(momentum=False)
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
parser.add_argument('--dont_save_images', dest='dont_save_images', action='store_true')
parser.set_defaults(dont_save_images=False)
parser.add_argument('--save_each_itr', dest='save_each_itr', action='store_true')
parser.set_defaults(save_each_itr=False)
hparams = parser.parse_args()

Pb = hparams.Pb
model_path = hparams.model_path
nb_itr = hparams.nb_itr
n_channels = hparams.n_channels
device = torch.device('cuda:'+str(hparams.gpu_number) if torch.cuda.is_available() else 'cpu')


def gen_data(clean_image, sigma, kernel, seed=0):
    """
    Generate the degradate observation
    """
    fft_k = deblur.p2o(kernel, clean_image.shape[-2:])
    temp = fft_k * deblur.fftn(clean_image)
    observation_without_noise = torch.abs(deblur.ifftn(temp))

    # For reproducibility
    gen = torch.Generator()
    gen.manual_seed(seed)
    noise = torch.normal(torch.zeros(observation_without_noise.size()), torch.ones(observation_without_noise.size()), generator = gen)*sigma / 255

    observation = observation_without_noise + noise
    return observation


# Parameters setting
denoiser_level = hparams.denoiser_level
lamb = hparams.lamb
sigma_obs = hparams.sigma_obs
r = hparams.r
B = hparams.B
theta = hparams.theta
Nesterov = hparams.Nesterov
momentum = hparams.momentum
restarting_su = hparams.restarting_su
restarting_li = hparams.restarting_li
stepsize = hparams.stepsize
dont_save_images = hparams.dont_save_images
save_each_itr = hparams.save_each_itr
alg = hparams.alg

Average_PSNR = 0

# Set input image paths
input_path = os.path.join('datasets', hparams.dataset_name)
input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

for i, clean_image_path in enumerate(input_paths):
    model = PnP(nb_itr, model_path, n_channels, device, Pb)
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()

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

    clean_image = util.imread_uint(clean_image_path, 3)
    clean_image = util.single2tensor3(clean_image).unsqueeze(0) /255.
    observation = gen_data(clean_image, sigma_obs, kernel)

    observation = observation.to(device)
    initial_uv = observation.clone()
    clean_image = clean_image.to(device)
    kernel = kernel.to(device)

    # Run RED algorithm with or without Nesterov momentum
    with torch.no_grad():
        model(initial_uv, observation, clean_image, kernel, sigma_obs, lamb, denoiser_level, theta, r, B, Nesterov, momentum, restarting_su, restarting_li, stepsize, alg)

    psnr_list = model.res['psnr']
    ssim_list = model.res['ssim']
    print("Restored image PSNR = {:.2f}".format(psnr_list[-1]))
    Average_PSNR += psnr_list[-1]
    if restarting_su or restarting_li:
        print("Number of restarting activation = {}".format(model.nb_restart_activ))
    
    savepth = 'results/'+hparams.dataset_name+"/"+alg+"_den_level_{}_lamb_{}".format(denoiser_level, lamb)
    os.makedirs(savepth, exist_ok = True)
    if '--kernel_index' in sys.argv:
        savepth = os.path.join(savepth, 'kernel_'+str(k_index))
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
    if '--B' in sys.argv:
        savepth = os.path.join(savepth, 'B_'+str(hparams.B))
        os.makedirs(savepth, exist_ok = True)
    if '--nb_itr' in sys.argv:
        savepth = os.path.join(savepth, 'nb_itr_'+str(hparams.nb_itr))
        os.makedirs(savepth, exist_ok = True)
    if '--stepsize' in sys.argv:
        savepth = os.path.join(savepth, 'stepsize_'+str(hparams.stepsize))
        os.makedirs(savepth, exist_ok = True)

    if not(dont_save_images):
        if save_each_itr:
            savepth_img = savepth+"/set_img_{}/".format(i)
            os.makedirs(savepth_img, exist_ok = True)
            for j in range(len(model.res['image'])):
                model.res['image'][j].save(savepth_img + 'iterations_{}.png'.format(j))
        
        model.res['image'][-1].save(savepth + '/restored_img.png')
        clean_img_uint = util.tensor2uint(clean_image)
        obs_uint = util.tensor2uint(observation)
        plt.imsave(savepth + '/clean_img.png', clean_img_uint)
        plt.imsave(savepth + '/observation.png', obs_uint)

        itr_list = range(len(psnr_list))
        plt.clf()
        plt.plot(itr_list, psnr_list, '-', alpha=0.8, linewidth=1.5)
        plt.xlabel('iter')
        plt.ylabel('PSNR')
        plt.savefig(savepth+'/PSNR_list.png')
        plt.clf()
        plt.plot(itr_list, ssim_list, '-', alpha=0.8, linewidth=1.5)
        plt.xlabel('iter')
        plt.ylabel('SSIM')
        plt.savefig(savepth+'/SSIM_list.png')
    
    dict = {
            'clean_image' : util.tensor2uint(clean_image),
            'observation' : util.tensor2uint(observation),
            'initial_uv' : initial_uv,
            'kernel' : util.tensor2uint(kernel),
            'sigma_obs' : sigma_obs,
            'lamb' : lamb,
            'denoiser_level' : denoiser_level,
            'r' : r,
            'Nesterov' : Nesterov,
            'restarting_su' : restarting_su,
            'stack_images' : model.res['image'],
            'clean_image_path' : clean_image_path,
            'kernel_path' : kernel_path,
            'psnr_list' : psnr_list,
            'ssim_list' : ssim_list,
            'psnr_restored' : psnr_list[-1],
            'ssim_restored' : ssim_list[-1],
            'restored' : model.res['image'][-1],
            'nb_restart_activ' : model.nb_restart_activ,
        }
    
    np.save(savepth+"/dict_results_{}".format(i), dict)

print("Average restored PSNR on the dataset = {:.2f}".format(Average_PSNR / len(input_paths)))
