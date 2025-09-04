for i in 1. 5.
do
    for j in 0.01 0.02 0.03 0.04 0.05
    do
        python main.py --sigma_obs 1. --stepsize 0.5 --nb_itr 500 --denoiser_level $j --lamb $i --gpu_number 1 --dataset_name "MRI_4knee" --denoiser_name "GSDRUNet_grayscale" --alg "GD" --Pb "MRI" --reduction_factor 8
    done
done

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




























# for i in 0.09 0.1 0.11
# do
#     for j in 13. 15. 18.
#     do 
#         python main.py --gpu_number 0 --dataset_name "set5" --stepsize 0.07 --momentum --theta 0.2 --lamb $j --denoiser_level $i --alg "GD" --Pb "deblurring"
#     done
# done






# for i in 0.09 0.1 0.11
# do
#     for j in 13.
#     do 
#         python main.py --gpu_number 0 --dataset_name "set5" --stepsize 0.1 --lamb $j --denoiser_level $i --alg "GD" --Pb "deblurring"
#     done
# done


###
# For ODT: 
###


# for i in  0.01 0.1 0.3   0.9   1.0
# do
#     CUDA_VISIBLE_DEVICES='0' python main_ODT.py --alg 'GD' --momentum --theta $i --lamb 1.5e6 --stepsize 1e-4
# # python generate_results.py
# done

# for i in 0.01 0.1 0.3   0.9  1.0
# do
#     for j in 100
#     do
#         CUDA_VISIBLE_DEVICES='0' python main_ODT.py --alg 'GD'  --theta $i --restarting_li --B $j --momentum --lamb 1.5e6 --stepsize 1e-4
#     done
# done

# for i in   0.5 1.0
# do
#     CUDA_VISIBLE_DEVICES='0' python main_ODT.py --alg 'PGD' --momentum --theta $i --lamb 3.75e3 --stepsize 2.5e-3
# # python generate_results.py
# done


# for i in   0.5 1.0
# do
#     for j in 100
#     do
#         CUDA_VISIBLE_DEVICES='0' python main_ODT.py --alg 'PGD' --theta $i --restarting_li --B $j --momentum --lamb 3.75e3 --stepsize 2.5e-3
#     done # python generate_results.py
# done

