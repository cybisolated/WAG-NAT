python script_train_tcn.py --dataset etth1 --data_dir data/ETT-small \
    --L_in 336 --L_out 168 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --d_model 128 --kernel_size 3 --n_levels 1 --hidden_size 256 --dropout 0.3 \
    --device cpu --num_workers 8 --batch_size 64