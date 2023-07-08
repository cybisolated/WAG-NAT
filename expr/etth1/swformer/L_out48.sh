python script_train_swformer.py --dataset etth1 --data_dir data/ETT-small \
    --L_in 48 --L_out 48 --patience 5 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_steps 2 \
    --valid_start 360 --test_start 480 \
    --d_model 256 --num_heads 2 --num_enc_layers 2 --num_dec_layers 1 \
    --window_size 8 --kernel_size 5 --dropout 0.3  --activation gelu \
    --device cpu --num_workers 8