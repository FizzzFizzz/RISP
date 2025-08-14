for i in 0.19
do
    for j in 20.
    do 
        python main.py --gpu_number 1 --dataset_name "set5" --stepsize 0.1 --lamb $j --denoiser_level $i --alg "PGD" --sigma_obs 25. --nb_itr 500
    done
done





"""
for i in 0 1 2 3 4 5 6 7 
do  
    for j in 0.01 0.1 0.2 0.3 0.5 0.9
    do
        python main.py --kernel_index $i --dataset_name "CBSD10" --stepsize 0.1 --theta $j --momentum
    done
done
"""

"""
for i in 0.01 0.1 0.2 0.3 0.4 0.6 0.9
do
    python main.py --alg 'PGD' --stepsize 2. --dataset_name "set5" --theta $i
    for j in 200 2000 5000 10000 50000
    do
        python main.py --alg 'PGD' --stepsize 2. --dataset_name "set5" --theta $i --restarting_li --B $j
    done
done
"""




# For ODT: 



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

