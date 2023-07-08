python script_tune_swformer.py --dataset ozone --data_dir data/ozone \
    --L_out 24 --patience 7 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 --lr_decay_steps 5 \
    --valid_start 1167 --test_start 1314 \
    --device cpu --num_workers 4 --batch_size 64 --num_trials 50

python script_tune_swformer.py --dataset ozone --data_dir data/ozone \
    --L_out 48 --patience 7 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 --lr_decay_steps 5 \
    --valid_start 1167 --test_start 1314 \
    --device cpu --num_workers 4 --batch_size 64 --num_trials 50

python script_tune_swformer.py --dataset ozone --data_dir data/ozone \
    --L_out 168 --patience 7 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 --lr_decay_steps 5 \
    --valid_start 1167 --test_start 1314 \
    --device cpu --num_workers 4 --batch_size 64 --num_trials 50

python script_tune_swformer.py --dataset ozone --data_dir data/ozone \
    --L_out 336 --patience 7 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 --lr_decay_steps 5 \
    --valid_start 1167 --test_start 1314 \
    --device cpu --num_workers 4 --batch_size 64 --num_trials 50

python script_tune_swformer.py --dataset ozone --data_dir data/ozone \
    --L_out 720 --patience 7 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_rate 0.1 --lr_decay_steps 5 \
    --valid_start 1167 --test_start 1314 \
    --device cpu --num_workers 4 --batch_size 64 --num_trials 50