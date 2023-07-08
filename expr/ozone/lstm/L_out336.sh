python script_train_lstm.py --dataset ozone --data_dir data/ozone \
    --L_in 336 --L_out 336 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 1167 --test_start 1314 \
    --d_model 256 --hidden_size 512 --num_layers 2 --dropout 0.2 \
    --device cpu --num_workers 8 --batch_size 64