python script_train_swformer.py --dataset etth2 --data_dir data/ETT-small \
    --L_in 24 --L_out 168 --patience 6 --loss mse --lr 0.0001 --lr_scheduler step\
    --valid_start 360 --test_start 480 \
    --d_model 32 --num_heads 8 --num_enc_layers 3 --num_dec_layers 1 \
    --window_size 8 --kernel_size 5 --dropout 0.30 --activation gelu \
    --device cpu --num_workers 8 --batch_size 32