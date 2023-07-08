python script_tune_tcn.py --dataset ettm1 --data_dir data/ETT-small \
    --L_out 24 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 0 --batch_size 32 --num_trials 50

python script_tune_tcn.py --dataset ettm1 --data_dir data/ETT-small \
    --L_out 48 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 0 --batch_size 32 --num_trials 50

python script_tune_tcn.py --dataset ettm1 --data_dir data/ETT-small \
    --L_out 96 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 0 --batch_size 32 --num_trials 50

python script_tune_tcn.py --dataset ettm1 --data_dir data/ETT-small \
    --L_out 288 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 0 --batch_size 32 --num_trials 50
