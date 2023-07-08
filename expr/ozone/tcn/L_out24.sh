python script_train_tcn.py --dataset ozone --data_dir data/ozone \
    --L_in 48 --L_out 24 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 1167 --test_start 1314 \
    --d_model 128 --kernel_size 5 --n_levels 4 --hidden_size 128 --dropout 0.1 \
    --device cpu --num_workers 8 --batch_size 64