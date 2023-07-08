python script_train_swformer.py --dataset ozone --data_dir data/ozone \
    --L_in 24 --L_out 24 --patience 5 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_steps 5 \
    --valid_start 1167 --test_start 1314 \
    --d_model 64 --num_heads 8 --num_enc_layers 1 --num_dec_layers 2 \
    --window_size 8 --kernel_size 5 --dropout 0.10 --activation gelu \
    --device cpu --num_workers 8 --batch_size 64