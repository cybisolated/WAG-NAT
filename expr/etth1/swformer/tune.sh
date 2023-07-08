python script_tune_swformer.py --dataset etth1 --data_dir data/ETT-small \
    --L_out 24 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 4 --batch_size 32 --num_trials 100

python script_tune_swformer.py --dataset etth1 --data_dir data/ETT-small \
    --L_out 48 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 4 --batch_size 32 --num_trials 100

python script_tune_swformer.py --dataset etth1 --data_dir data/ETT-small \
    --L_out 168 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 4 --batch_size 32 --num_trials 100

python script_tune_swformer.py --dataset etth1 --data_dir data/ETT-small \
    --L_out 336 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 4 --batch_size 32 --num_trials 100

python script_tune_swformer.py --dataset etth1 --data_dir data/ETT-small \
    --L_out 720 --patience 8 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 4 --batch_size 32 --num_trials 100