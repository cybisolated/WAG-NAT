python script_train_informer.py --dataset ozone --data_dir data/ozone \
    --L_in 168 --L_token 168 --L_out 168 --patience 5 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1\
    --valid_start 1167 --test_start 1314 \
    --d_model 512 --n_heads 8 --d_ff 2048 --e_layers 2 --d_layers 1 --attn prob --factor 5 \
    --device cuda --num_workers 8 --batch_size 64