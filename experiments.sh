python main.py --gpu_number 0 --dataset_name "CBSD68" --stepsize 2.0  --lamb 15. --denoiser_level 0.1 --alg "PGD" --Pb "deblurring"

python main.py --gpu_number 0 --dataset_name "CBSD68" --stepsize 0.1 --lamb 15. --denoiser_level 0.1 --alg "GD" --Pb "deblurring"

for i in 0.01 0.1 0.2 0.3 0.9
do
    python main.py --gpu_number 0 --dataset_name "CBSD68" --stepsize 0.07 --momentum --theta 0.2 --lamb 15. --denoiser_level 0.1 --alg "GD" --Pb "deblurring" --restarting_li

    python main.py --gpu_number 0 --dataset_name "CBSD68" --stepsize 0.07 --momentum --theta 0.2 --lamb 15. --denoiser_level 0.1 --alg "GD" --Pb "deblurring"

    python main.py --gpu_number 0 --dataset_name "CBSD68" --stepsize 5.0 --momentum --theta 0.2 --lamb 15. --denoiser_level 0.1 --alg "PGD" --Pb "deblurring" --restarting_li

    python main.py --gpu_number 0 --dataset_name "CBSD68" --stepsize 5.0 --momentum --theta 0.2 --lamb 15. --denoiser_level 0.1 --alg "PGD" --Pb "deblurring"
done

# for i in 0.09 0.1 0.11
# do
#     for j in 13. 15. 18.
#     do 
#         python main.py --gpu_number 0 --dataset_name "set5" --stepsize 0.07 --momentum --theta 0.2 --lamb $j --denoiser_level $i --alg "GD" --Pb "deblurring"
#     done
# done

# for i in 0.06 0.08 0.10
# do
#     for j in 3.
#     do 
#         python main.py --gpu_number 0 --dataset_name "set5" --stepsize 5. --nb_itr 500 --sigma_obs 1.0 --p 0.2 --lamb $j --denoiser_level $i --alg "PGD" --Pb "inpainting"
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

