for i in 100 1000 3000 5000 7000 10000 100000
do  
    python main.py --dont_save_images --reduction_factor 8 --sigma_obs 1.0 --stepsize 0.1 --denoiser_level 0.02 --nb_itr 500 --lamb 1.0 --gpu_number 0 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --alg "GD" --Pb "MRI" --momentum --theta 0.2 --restarting_li --B $i
    python main.py --dont_save_images --reduction_factor 8 --sigma_obs 1.0 --stepsize 0.1 --denoiser_level 0.02 --nb_itr 500 --lamb 1.0 --gpu_number 0 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --alg "GD" --Pb "MRI" --momentum --theta 0.001 --restarting_li --B $i
done
python main.py --dont_save_images --reduction_factor 8 --sigma_obs 1.0 --stepsize 0.1 --denoiser_level 0.02 --nb_itr 500 --lamb 1.0 --gpu_number 0 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --alg "GD" --Pb "MRI" --momentum --theta 0.2
python main.py --dont_save_images --reduction_factor 8 --sigma_obs 1.0 --stepsize 0.1 --denoiser_level 0.02 --nb_itr 500 --lamb 1.0 --gpu_number 0 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --alg "GD" --Pb "MRI" --momentum --theta 0.001


# python main.py --start_im_indx 47 --sigma_obs 1.0 --save_frequency 100 --dont_save_images --nb_itr 1500 --stepsize 5. --denoiser_level 0.08 --lamb 5. --gpu_number 1 --p 0.2 --dataset_name "CBSD68" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "inpainting" --momentum --theta 0.2 --restarting_li


# python main.py --sigma_obs 1.0 --save_frequency 100 --dont_save_images --nb_itr 1500 --stepsize .1 --denoiser_level 0.08 --lamb 5. --gpu_number 1 --p 0.2 --dataset_name "CBSD68" --denoiser_name "GSDRUNet_SoftPlus" --alg "GD" --Pb "inpainting" --momentum --theta 0.2 --restarting_li
# python main.py --sigma_obs 1.0 --save_frequency 100 --dont_save_images --nb_itr 1500 --stepsize .1 --denoiser_level 0.08 --lamb 5. --gpu_number 0 --p 0.2 --dataset_name "CBSD68" --denoiser_name "GSDRUNet_SoftPlus" --alg "GD" --Pb "inpainting"
# python main.py --sigma_obs 1.0 --save_frequency 100 --dont_save_images --nb_itr 1500 --stepsize 5. --denoiser_level 0.08 --lamb 5. --gpu_number 1 --p 0.2 --dataset_name "CBSD68" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "inpainting" --momentum --theta 0.2 --restarting_li
# python main.py --sigma_obs 1.0 --save_frequency 100 --dont_save_images --nb_itr 1500 --stepsize 5. --denoiser_level 0.08 --lamb 5. --gpu_number 0 --p 0.2 --dataset_name "CBSD68" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "inpainting"




# for i in 0
# do  
#     python main.py --sigma_obs 1.0 --dont_save_images --kernel_index $i --stepsize 0.4 --denoiser_level 0.03 --nb_itr 250 --lamb 10. --gpu_number 0 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "GD" --Pb "SR" --sf 2
#     # python main.py --sigma_obs 1.0 --dont_save_images --kernel_index $i --stepsize 0.4 --denoiser_level 0.03 --nb_itr 250 --lamb 10. --gpu_number 0 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "GD" --Pb "SR" --sf 2 --momentum --theta 0.2 --restarting_li
#     # python main.py --sigma_obs 1.0 --dont_save_images --kernel_index $i --stepsize 10.0 --denoiser_level 0.03 --nb_itr 250 --lamb 10. --gpu_number 1 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "SR" --sf 2 --momentum --theta 0.2 --restarting_li
#     # python main.py --sigma_obs 1.0 --dont_save_images --kernel_index $i --stepsize 10.0 --denoiser_level 0.03 --nb_itr 250 --lamb 10. --gpu_number 1 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "SR" --sf 2
# done




# python main.py --sigma_obs 1.0 --dont_save_images --reduction_factor 8 --nb_itr 500 --stepsize 0.1 --denoiser_level 0.02 --lamb 1. --gpu_number 1 --dataset_name "MRI_knee" --alg "GD" --Pb "MRI" --denoiser_name "GSDRUNet_grayscale" --momentum --theta 0.2 --restarting_li
# python main.py --sigma_obs 1.0 --dont_save_images --reduction_factor 8 --nb_itr 500 --stepsize 0.5 --denoiser_level 0.02 --lamb 1. --gpu_number 1 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --alg "PGD" --Pb "MRI"









# python main.py --sigma_obs 1. --dont_save_images --stepsize 0.7 --nb_itr 500 --denoiser_level 0.01 --lamb 1. --gpu_number 0 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --alg "GD" --Pb "MRI" --reduction_factor 4
# python main.py --sigma_obs 1. --dont_save_images --stepsize 0.7 --nb_itr 500 --denoiser_level 0.01 --lamb 1. --gpu_number 0 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --momentum --theta 0.2 --restarting_li --alg "GD" --Pb "MRI" --reduction_factor 4
# python main.py --sigma_obs 1. --dont_save_images --stepsize 2.0 --nb_itr 500 --denoiser_level 0.01 --lamb 1. --gpu_number 1 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --alg "PGD" --Pb "MRI" --reduction_factor 4
# python main.py --sigma_obs 1. --dont_save_images --stepsize 1.0 --nb_itr 500 --denoiser_level 0.01 --lamb 1. --gpu_number 1 --dataset_name "MRI_knee" --denoiser_name "GSDRUNet_grayscale" --momentum --theta 0.2 --restarting_li --alg "PGD" --Pb "MRI" --reduction_factor 4


# for i in 0 9 
# do
    # python main.py --dont_save_images --nb_itr 500 --sigma_obs 1. --kernel_index $i --stepsize 0.7 --denoiser_level 0.03 --lamb 10. --gpu_number 1 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "GD" --Pb "SR"
    # python main.py --dont_save_images --nb_itr 500 --sigma_obs 1. --kernel_index $i --stepsize 0.4 --denoiser_level 0.03 --lamb 10. --gpu_number 1 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "GD" --momentum --theta 0.2 --restarting_li --Pb "SR"
    # python main.py --dont_save_images --nb_itr 500 --sigma_obs 1. --kernel_index $i --stepsize 10.0 --denoiser_level 0.03 --lamb 10. --gpu_number 0 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "SR"
    # python main.py --dont_save_images --nb_itr 500 --sigma_obs 1. --kernel_index $i --stepsize 10.0 --denoiser_level 0.03 --lamb 10. --gpu_number 0 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --momentum --theta 0.2 --restarting_li --Pb "SR"
# done

# python main.py --sigma_obs 1. --stepsize 5.0 --nb_itr 50 --denoiser_level 0.08 --lamb 5. --gpu_number 1 --dataset_name "CBSD68" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "inpainting" --p 0.2


# do
#     for j in 0.01 0.02 0.03 0.04 0.05
#     do
#         python main.py --sigma_obs 1. --stepsize 0.5 --nb_itr 500 --denoiser_level $j --lamb $i --gpu_number 1 --dataset_name "MRI_4knee" --denoiser_name "GSDRUNet_grayscale" --alg "GD" --Pb "MRI" --reduction_factor 8
#     done
# done

# python main.py --gpu_number 1 --nb_itr 10 --dataset_name "SAR" --denoiser_name "GSDRUNet_grayscale" --stepsize 0.01 --lamb 150. --denoiser_level 0.2 --alg "GD" --Pb "speckle" --L 10 --grayscale



# python main.py --gpu_number 1 --nb_itr 9 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --dont_save_images --stepsize 0.07 --momentum --theta 0.2 --lamb 15. --denoiser_level 0.1 --alg "GD" --Pb "deblurring" --restarting_li --kernel_index 0
# python main.py --gpu_number 1 --nb_itr 9 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --dont_save_images --stepsize 2.0 --lamb 15. --denoiser_level 0.1 --alg "PGD" --Pb "deblurring" --kernel_index 0
# python main.py --gpu_number 1 --nb_itr 9 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --dont_save_images --stepsize 5.0 --momentum --theta 0.2 --lamb 15. --denoiser_level 0.1 --alg "PGD" --Pb "deblurring" --restarting_li --kernel_index 0

# for b in 100.0 1000.0 3000.0 5000.0 7000.0 10000.0 100000.0
# do
#     python main.py --gpu_number 1 --B $b --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --dont_save_images --stepsize 0.07 --momentum --theta 0.01 --lamb 15. --denoiser_level 0.1 --alg "GD" --Pb "deblurring" --restarting_li --kernel_index 0
# done




# for k in 0 1 2 3 4 5 6 7 8 9
# do
#     for i in 0.01 0.1 0.3 0.9
#     do
#         python main.py --gpu_number 1 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --dont_save_images --stepsize 0.07 --momentum --theta $i --lamb 15. --denoiser_level 0.1 --alg "GD" --Pb "deblurring" --restarting_li --kernel_index $k
#         python main.py --gpu_number 1 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --dont_save_images --stepsize 0.07 --momentum --theta $i --lamb 15. --denoiser_level 0.1 --alg "GD" --Pb "deblurring" --kernel_index $k
#         python main.py --gpu_number 1 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --dont_save_images --stepsize 5.0 --momentum --theta $i --lamb 15. --denoiser_level 0.1 --alg "PGD" --Pb "deblurring" --restarting_li --kernel_index $k
#         python main.py --gpu_number 1 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --dont_save_images --stepsize 5.0 --momentum --theta $i --lamb 15. --denoiser_level 0.1 --alg "PGD" --Pb "deblurring" --kernel_index $k
#     done
# done









### For Rician noise removal.
# RISP-GM
python main.py --Pb "rician" --sigma_obs 25.5 --save_frequency 5  --dataset_name 'CBSD10' --denoiser_name "GSDRUNet_SoftPlus" --nb_itr 500 --lamb 5e-3 --denoiser_level 0.05 --stepsize 0.03 --theta 0.01 --restarting_li --B 1000 --momentum --alg "GD" --dont_compute_potential
# RED-GM
python main.py --Pb "rician" --sigma_obs 25.5 --save_frequency 5  --dataset_name 'CBSD10' --denoiser_name "GSDRUNet_SoftPlus" --nb_itr 500 --lamb 5e-3 --denoiser_level 0.05 --stepsize 0.03 --theta 1.00 --restarting_li --B 1e10 --momentum --alg "GD" --dont_compute_potential
# RISP-Prox
python main.py --Pb "rician" --sigma_obs 25.5 --save_frequency 5  --dataset_name 'CBSD10' --denoiser_name "GSDRUNet_SoftPlus" --nb_itr 500 --lamb 5e-3 --denoiser_level 0.05 --stepsize 5e-4 --theta 0.01 --restarting_li --B 1000 --momentum --alg "PGD" --dont_compute_potential
# RED-Prox
python main.py --Pb "rician" --sigma_obs 25.5 --save_frequency 5  --dataset_name 'CBSD10' --denoiser_name "GSDRUNet_SoftPlus" --nb_itr 500 --lamb 5e-3 --denoiser_level 0.05 --stepsize 5e-4 --theta 1.00 --restarting_li --B 1e10 --momentum --alg "PGD" --dont_compute_potential

### For linear inverse scattering with image size 1024, 360 rec, 240 trans, 0.0001 noise:
# RISP-GM
python main.py --Pb "ODT" --save_frequency 2000 --ODT_Nxy 1024 --ODT_Rec 360 --ODT_Trans 240 --dataset_name 'ODT1024_01' --denoiser_name "DRUNet_ODT" --nb_itr 20000 --lamb 1e5 --denoiser_level 0.03 --stepsize 1e-3 --theta 0.01 --restarting_li --B 5e5 --momentum --alg "GD" 
# RED-GM
python main.py --Pb "ODT" --save_frequency 2000 --ODT_Nxy 1024 --ODT_Rec 360 --ODT_Trans 240 --dataset_name 'ODT1024_01' --denoiser_name "DRUNet_ODT" --nb_itr 20000 --lamb 1e5 --denoiser_level 0.03 --stepsize 1e-3 --theta 1.00 --restarting_li --B 1e10 --momentum --alg "GD" 
# RISP-Prox
python main.py --Pb "ODT" --save_frequency 2000 --ODT_Nxy 1024 --ODT_Rec 360 --ODT_Trans 240 --dataset_name 'ODT1024_01' --denoiser_name "DRUNet_ODT" --nb_itr 10000 --lamb 1e5 --denoiser_level 0.03 --stepsize 0.005 --theta 0.01 --restarting_li --B 5e5 --momentum --alg "PGD" 
# RISP-GM
python main.py --Pb "ODT" --save_frequency 2000 --ODT_Nxy 1024 --ODT_Rec 360 --ODT_Trans 240 --dataset_name 'ODT1024_01' --denoiser_name "DRUNet_ODT" --nb_itr 10000 --lamb 1e5 --denoiser_level 0.03 --stepsize 0.005 --theta 1.00 --restarting_li --B 1e10 --momentum --alg "PGD" 







