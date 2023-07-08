python script_train_conv_trans.py --dataset ozone --data_dir data/ozone \
    --L_in 168 --L_out 168 --patience 5 --loss mse --lr 0.0001 --lr_scheduler step \
    --valid_start 1167 --test_start 1314 \
    --d_model 512 --num_heads 4 --num_layers 3 --q_len 3 --sparse_attn --sub_len 1 --dropout 0.20 \
    --device cpu --num_workers 8 --batch_size 64