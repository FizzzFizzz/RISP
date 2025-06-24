for i in 2. 1. 0.5
do
    python main.py --alg 'PGD' --stepsize $i --dataset_name "set5"
done





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