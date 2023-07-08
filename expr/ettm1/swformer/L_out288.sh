python script_train_swformer.py --dataset ettm1 --data_dir data/ETT-small \
    --L_in 336 --L_out 288 --patience 5 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_steps 5 \
    --valid_start 360 --test_start 480 \
    --d_model 64 --num_heads 4 --num_enc_layers 3 --num_dec_layers 2 \
    --window_size 8 --kernel_size 5 --dropout 0.2  --activation gelu \
    --device cpu --num_workers 8