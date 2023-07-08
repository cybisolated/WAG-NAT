import pickle
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
import math
import os
import torch
from argparse import ArgumentParser
from dataset import TimeSeriesDataset
from model import SWFormer
from script_get_data import *
from data_formatters import OzoneFormatter, ElectricityFormatter, ETTFormatter
from torch.utils.data import DataLoader
from model import PlModelInterface
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from libs.utils import get_embedding_size, load_dataset

from vis_res import vis_res
from libs.utils import create_folder_if_not_exist, get_embedding_size


def objective(trial: optuna.trial.Trial):
    torch.multiprocessing.set_sharing_strategy('file_system')
    pl.seed_everything(args.seed)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'

    # load processed data (pd.DataFrame) and create DataFormatter
    data_dir = args.data_dir
    dataset_name = args.dataset
    df_data, formatter = load_dataset(data_dir, dataset_name)

    # split data to train, valid and test
    L_in = trial.suggest_categorical('L_in', [24, 48, 168, 336]) if args.L_in is None else args.L_in
    L_out = args.L_out

    valid_start, test_start = args.valid_start, args.test_start
    df_train, df_valid, df_test = formatter.split_data(df_data, L_in, valid_start=valid_start, test_start=test_start)

    # construct train/val/test Dataset
    input_names = formatter.get_input_names()
    train_samples, valid_samples = args.train_samples, args.valid_samples
    train_set, valid_set, test_set = [
        TimeSeriesDataset(
            d, L_in, L_out, **input_names, num_samples=train_samples if i == 0 else (valid_samples if i == 1 else None)
        )
        for i, d in enumerate([df_train, df_valid, df_test])
    ]

    print(f'number of train samples: {len(train_set)}')
    print(f'number of valid samples: {len(valid_set)}')
    print(f'number of test samples: {len(test_set)}')

    batch_size = args.batch_size
    num_workers = args.num_workers
    persistent_workers = args.persistent_workers
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )

    num_classes = formatter.num_classes_per_cat_input

    model_params = dict(
        L_in=L_in,
        L_out=L_out,
        num_classes=num_classes,
        d_model=trial.suggest_categorical('d_model', [32, 64, 128, 256, 512]),
        window_size=trial.suggest_categorical('window_size', [4, 6, 8, 12]),
        kernel_size=trial.suggest_categorical('kernel_size', [3, 5, 7]),
        num_heads=trial.suggest_categorical('num_heads', [1, 2, 4, 8]),
        num_enc_layers=trial.suggest_int('num_enc_layers', 1, 3),
        num_dec_layers=trial.suggest_int('num_decoder_layers', 1, 3),
        activation='gelu',
        dropout=trial.suggest_categorical('dropout', [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]),
        **input_names,
    )

    swformer = SWFormer(**model_params)

    logger = TensorBoardLogger(save_dir=os.path.join(args.save_dir, dataset_name), name=args.model_name)
    callbacks = [
        EarlyStopping('val_loss', min_delta=args.min_delta, patience=args.patience, mode='min'),
        PyTorchLightningPruningCallback(trial, monitor='val_loss'),
        LearningRateMonitor(),
    ]

    model = PlModelInterface(model=swformer, **vars(args))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.device,
        enable_checkpointing=False,
        enable_model_summary=True,
        callbacks=callbacks,
        logger=[logger],
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=0,
    )

    trial_params = trial.params

    # with open(os.path.join(args.study_dir, f'study_{study_name}_trial{trial.number}_params.pkl'), 'wb') as f:
    #     pickle.dump(trial_params, f)
    print(trial_params)

    trainer.fit(model, train_loader, valid_loader)

    return model.best_val_loss


parser = ArgumentParser()

parser.add_argument('--tune', default=True, action='store_true')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--max_epochs', default=100, type=int)
parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--val_check_interval', default=1.0, type=float)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--persistent_workers', action='store_true')
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--min_delta', default=1e-4, type=float)

# dataset params
parser.add_argument('--dataset', default='ozone', type=str)
parser.add_argument('--data_dir', default='data/ozone', type=str)
parser.add_argument('--train_samples', default=None, type=int)
parser.add_argument('--valid_samples', default=None, type=int)

# tensorboard params
parser.add_argument('--save_dir', default='./tuning_logs', type=str)
parser.add_argument('--model_name', default='swformer', type=str)

parser.add_argument('--lr_scheduler', choices=['cosine', 'step'], default=None, type=str)
parser.add_argument('--lr_decay_steps', default=5, type=int)
parser.add_argument('--lr_decay_rate', default=0.1, type=float)
parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

parser.add_argument('--loss', choices=['mse', 'L1', 'smooth_L1', 'huber'], default='mse', type=str)
parser.add_argument('--huber_delta', default=1.35, type=float)
parser.add_argument('--L_in', default=None, type=int)
parser.add_argument('--L_out', default=24, type=int)
parser.add_argument('--valid_start', default=900, type=int)
parser.add_argument('--test_start', default=1200, type=int)

parser.add_argument('--study_dir', default='./study', type=str)
parser.add_argument('--num_trials', default=100, type=int)

args = parser.parse_args()

args.study_dir = os.path.join(args.study_dir, args.dataset, args.model_name)
create_folder_if_not_exist(args.study_dir)

if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner()

    study_name = f'{args.model_name}_{args.dataset}_tune_L_out={args.L_out}'
    study = optuna.create_study(study_name=study_name, direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=args.num_trials, timeout=None)

    with open(os.path.join(args.study_dir, f'{study_name}.pkl'), 'wb') as f:
        pickle.dump(study, f)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(os.path.join(args.study_dir, f'study_{study_name}_best_params.pkl'), 'wb') as f:
        pickle.dump(trial.params, f)
