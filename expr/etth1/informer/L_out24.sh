python script_train_informer.py --dataset etth1 --data_dir data/ETT-small \
    --L_in 48 --L_token 48 --L_out 24 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --d_model 512 --n_heads 8 --d_ff 2048 --e_layers 2 --d_layers 1 --attn prob --factor 3 \
    --device cuda --num_workers 8 --batch_size 64