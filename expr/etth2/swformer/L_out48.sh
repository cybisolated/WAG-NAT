python script_train_swformer.py --dataset etth2 --data_dir data/ETT-small \
    --L_in 48 --L_out 48 --patience 6 --loss mse --lr 0.0001 --lr_scheduler step\
    --valid_start 360 --test_start 480 \
    --d_model 128 --num_heads 2 --num_enc_layers 2 --num_dec_layers 1 \
    --window_size 8 --kernel_size 7 --dropout 0.20 --activation gelu \
    --device cpu --num_workers 8 --batch_size 32