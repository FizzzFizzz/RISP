for i in 15 18
do
    python main.py --dataset_name "CBSD10" --stepsize 5.0 --alg "PGD" --momentum --theta 0.2 --lamb $i
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