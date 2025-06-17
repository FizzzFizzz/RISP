for i in 0 1 2 3 4 5 6 7 8 9
    do
        for j in 0.01 0.05 0.1 0.2
        do
            python main.py --dataset_name "CBSD10" --dont_save_images --momentum --theta $j --kernel_index $i
        done
        python main.py --dataset_name "CBSD10" --dont_save_images --kernel_index $i
        for j in 0.01 0.05 0.1 0.2
        do
            for k in 100 1000 5000 10000
            do
            python main.py --dataset_name "CBSD10" --dont_save_images --momentum --theta $j --kernel_index $i --restarting_li --B $k
            done
        done
    done

