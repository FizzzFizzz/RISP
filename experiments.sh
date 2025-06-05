for j in 0.01 0.05 0.1 0.2
    do
        python main.py --dataset_name "set5" --momentum --theta $j --dont_save_images --nb_itr 200
    done
