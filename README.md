# RISP

This repos contains the code associated with the paper "Provably Accelerated Imaging with Restarted Inertia and Score-based image priors" accepted by ICLR 2026.

## Prerequisites - setting the environment

To create a conda environment with the correct dependency to use the code, run in your terminal 
```
conda env create -f environment.yml
```

## Pre-trained weights
This repos is using pre-trained weights that need to be download and store in the folder "model_ckpt" namely
- DRUNet learned on color natural images in this [link](https://huggingface.co/deepinv/drunet/resolve/main/drunet_color.pth?download=true)
- GSDRUNet learned on color natural images with SoftPlus activations in this [link](https://plmbox.math.cnrs.fr/f/414fbb3e754840978ef8/?dl=1)
- GSDRUNet in grayscale natural images in this [link](https://huggingface.co/deepinv/gradientstep/resolve/main/GSDRUNet_grayscale_torch.ckpt) 
- GSDRUNet learned on color natural images in this [link](https://huggingface.co/deepinv/gradientstep/resolve/main/GSDRUNet.ckpt)


## How to use the code - some example

We present here, some examples of use of the code. Please note that the hyperparameters are not hard-coding in this repos. Please refer to Table 1 in the paper for the suggested hyperparameters choices.

In order, to run the standard RED-GM algorithm on deblurring run
```
python main.py --lamb 15.0 --denoiser_level 0.1 --stepsize 0.1
```

In order, to run the RISP-GM algorithm on inpainting run
```
python main.py --Pb "inpainting" --alg "GD" --restarting_li --momentum --lamb 5.0 --denoiser_level 0.08 --stepsize 0.1
```

In order, to run the RISP-Prox algorithm on Rician noise removal run
```
python main.py --Pb "rician" --alg "PGD" --restarting_li --momentum --lamb 0.005 --denoiser_level 0.05 --stepsize 0.0005 --theta 0.01 --B 100
```

## Acknowledgement

This repo contains part of code from
- [DPIR](https://github.com/cszn/DPIR) the code associated with this [paper](https://arxiv.org/pdf/2008.13751).
- [GSPnP](https://github.com/samuro95/gspnp) the code associated with this [paper](https://arxiv.org/abs/2110.03220)
- [BCRED](https://github.com/hdocmsu/bcred) the code associated with this [paper](https://arxiv.org/abs/1905.05113)
- [SNORE](https://github.com/marien-renaud/snore) the code associated with this [paper](https://arxiv.org/abs/2402.01779)
