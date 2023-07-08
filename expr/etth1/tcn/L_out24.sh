python script_train_tcn.py --dataset etth1 --data_dir data/ETT-small \
    --L_in 48 --L_out 24 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --d_model 128 --kernel_size 5 --n_levels 4 --hidden_size 128 --dropout 0.1 \
    --device cpu --num_workers 8 --batch_size 64