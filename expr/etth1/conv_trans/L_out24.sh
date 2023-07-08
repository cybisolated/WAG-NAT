python script_train_conv_trans.py --dataset etth1 --data_dir data/ETT-small \
    --L_in 96 --L_out 24 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step\
    --valid_start 360 --test_start 480 \
    --d_model 40 --num_heads 2 --num_layers 2 --q_len 3 --sparse_attn --sub_len 1 --dropout 0.1 \
    --device cpu --num_workers 4 --batch_size 32