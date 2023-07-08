# python script_tune_swformer.py --dataset etth2 --data_dir data/ETT-small \
#     --L_out 24 --patience 6 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_steps 5 \
#     --valid_start 360 --test_start 480 \
#     --device cpu --num_workers 8 --batch_size 32 --num_trials 70

# python script_tune_swformer.py --dataset etth2 --data_dir data/ETT-small \
#     --L_out 48 --patience 6 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_steps 5 \
#     --valid_start 360 --test_start 480 \
#     --device cpu --num_workers 8 --batch_size 32 --num_trials 70

python script_tune_swformer.py --dataset etth2 --data_dir data/ETT-small \
    --L_out 168 --patience 6 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_steps 5 \
    --valid_start 360 --test_start 480 \
    --device cpu --num_workers 8 --batch_size 32 --num_trials 70

# python script_tune_swformer.py --dataset etth2 --data_dir data/ETT-small \
#     --L_out 336 --patience 6 --loss mse --lr 0.0001 --lr_scheduler step --lr_decay_steps 5 \
#     --valid_start 360 --test_start 480 \
#     --device cpu --num_workers 8 --batch_size 32 --num_trials 70
