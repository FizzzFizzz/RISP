for i in 100 1000 3000 5000 7000 10000 100000
do 
    python main.py --dataset_name "set5" --momentum --theta 0.05 --restarting_li --B $i --dont_save_images
done