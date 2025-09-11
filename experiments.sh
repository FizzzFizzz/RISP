python main.py --sigma_obs 1. --stepsize 5.0 --nb_itr 50 --denoiser_level 0.08 --lamb 5. --gpu_number 1 --dataset_name "CBSD68" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "inpainting" --p 0.2


# for i in 5 6 7 8 9 
# do
#     python main.py --sigma_obs 1. --dont_save_images --kernel_index $i --stepsize 0.7 --nb_itr 500 --denoiser_level 0.03 --lamb 10. --gpu_number 0 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "GD" --Pb "SR" --sf 2
#     python main.py --sigma_obs 1. --dont_save_images --kernel_index $i --stepsize 10.0 --nb_itr 500 --denoiser_level 0.03 --lamb 10. --gpu_number 0 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --Pb "SR" --sf 2
#     python main.py --sigma_obs 1. --dont_save_images --kernel_index $i --stepsize 0.4 --nb_itr 500 --denoiser_level 0.03 --lamb 10. --gpu_number 0 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "GD" --momentum --theta 0.2 --restarting_li --Pb "SR" --sf 2
#     python main.py --sigma_obs 1. --dont_save_images --kernel_index $i --stepsize 10.0 --nb_itr 500 --denoiser_level 0.03 --lamb 10. --gpu_number 0 --dataset_name "CBSD10" --denoiser_name "GSDRUNet_SoftPlus" --alg "PGD" --momentum --theta 0.2 --restarting_li --Pb "SR" --sf 2
# done



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



# python main.py --save_frequency 10 --Pb "ODT" --ODT_Nxy 128 --ODT_Rec 180 --ODT_Trans 20 --dataset_name 'ODT10' --denoiser_name "DRUNet_ODT" --nb_itr 50 --denoiser_level 0.03 --stepsize 4e-3 --lamb 1e5 --alg "GD" 







#### For ODT with image size 1024, 360 rec, 240 trans, 0.0001 noise:

# # RiRED
# for i in 0.01  1.0
# do
#     python main.py --Pb "ODT" --save_frequency 200 --ODT_Nxy 1024 --ODT_Rec 360 --ODT_Trans 240 --dataset_name 'ODT1024_01' --denoiser_name "DRUNet_ODT" --nb_itr 20000 --lamb 1e5 --denoiser_level 0.03 --stepsize 4e-3 --theta $i --restarting_li --B 5e5 --momentum --alg "GD" 
# done

# # Prox-RiRED
# for i in 0.01  1.0
# do
#     python main.py --Pb "ODT" --save_frequency 200 --ODT_Nxy 1024 --ODT_Rec 360 --ODT_Trans 240 --dataset_name 'ODT1024_01' --denoiser_name "DRUNet_ODT" --nb_itr 10000 --lamb 1e5 --denoiser_level 0.03 --stepsize 0.005 --theta $i --restarting_li --B 5e5 --momentum --alg "PGD" 
# done





#### For Rician noise removal on natural color images, noise level 25.5
##### RiRED
# for i in 0.01  1.0  
# do
#     for B in 1e3
#     do
#         python main.py --Pb "rician" --sigma_obs 25.5 --save_frequency 5  --dataset_name 'CBSD68_cut8_10' --denoiser_name "GSDRUNet_SoftPlus" --nb_itr 500 --lamb 5e-3 --denoiser_level 0.05 --stepsize 0.03 --theta $i --restarting_li --B $B --momentum --alg "GD" 
#     done
# done

# ##### Prox-RiRED
# for i in  1.0 0.01 
# do
#     for B in 1e3 
#     do
#         python main.py --Pb "rician" --sigma_obs 25.5 --save_frequency 5  --dataset_name 'CBSD68_cut8_10' --denoiser_name "GSDRUNet_SoftPlus" --nb_itr 500 --lamb 5e-3 --denoiser_level 0.05 --stepsize 5e-4 --theta $i --restarting_li --B $B --momentum --alg "PGD" 
#     done
# done









