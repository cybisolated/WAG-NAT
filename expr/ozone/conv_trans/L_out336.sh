python script_train_conv_trans.py --dataset ozone --data_dir data/ozone \
    --L_in 336 --L_out 336 --patience 5 --loss mse --lr 0.0001 --lr_scheduler step \
    --valid_start 1167 --test_start 1314 \
    --d_model 256 --num_heads 8 --num_layers 3 --q_len 5 --sparse_attn --sub_len 1 --dropout 0.20 \
    --device cpu --num_workers 8 --batch_size 64