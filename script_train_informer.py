import math
import os
import torch
from argparse import ArgumentParser
from dataset import TimeSeriesDataset
from model import Informer, PlModelInterface
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from libs.utils import load_dataset


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    pl.seed_everything(args.seed)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    if torch.cuda.is_available():
        args.device = 'cuda'

    # load processed data (pd.DataFrame)
    data_dir = args.data_dir
    dataset_name = args.dataset
    df_data, formatter = load_dataset(data_dir, dataset_name)

    # split data to train, valid and test
    L_in, L_out = args.L_in, args.L_out
    valid_start, test_start = args.valid_start, args.test_start
    df_train, df_valid, df_test = formatter.split_data(df_data, L_in, valid_start=valid_start, test_start=test_start)

    # construct train/val/test Dataset
    input_names = formatter.get_input_names()
    train_set, valid_set, test_set = [
        TimeSeriesDataset(
            d,
            L_in,
            L_out,
            **input_names,
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
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )

    num_classes = formatter.num_classes_per_cat_input
    args.num_classes = num_classes
    informer = Informer(**vars(args), **input_names)

    logger = TensorBoardLogger(save_dir=os.path.join(args.save_dir, dataset_name), name=args.model_name)
    callbacks = [
        EarlyStopping('val_loss', min_delta=args.min_delta, patience=args.patience, mode='min'),
        LearningRateMonitor(),
        ModelCheckpoint(
            monitor='val_loss', filename='best-{epoch:03d}-{val_loss}', save_top_k=1, mode='min', save_last=True
        ),
    ]

    version = logger.version
    args.test_save_dir = os.path.join(args.test_save_dir, dataset_name, args.model_name, f'version_{version}')
    model = PlModelInterface(model=informer, **vars(args))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.device,
        enable_model_summary=True,
        callbacks=callbacks,
        logger=[logger],
        val_check_interval=args.val_check_interval,
    )

    if not args.only_test:
        trainer.fit(model, train_loader, valid_loader)

    if args.test:
        trainer.test(model, test_loader, ckpt_path='best')
        # vis_res(os.path.join(args.test_save_dir, 'test_outputs.pickle'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--only_test', action='store_true', default=False)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--min_delta', default=1e-4, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--val_check_interval', default=1.0, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    parser.add_argument('--test_save_dir', default='./output', type=str)

    # dataset params
    parser.add_argument('--dataset', default='ozone', type=str)
    parser.add_argument('--data_dir', default='data/ozone', type=str)

    # tensorboard params
    parser.add_argument('--save_dir', default='./training_logs', type=str)
    parser.add_argument('--model_name', default='informer', type=str)

    parser.add_argument('--lr_scheduler', choices=['cosine', 'step'], default=None, type=str)
    parser.add_argument('--lr_decay_steps', default=15, type=int)
    parser.add_argument('--lr_decay_rate', default=0.4, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-6, type=float)

    parser.add_argument('--loss', choices=['mse', 'L1', 'smooth_L1'], default='mse', type=str)
    parser.add_argument('--L_in', default=48, type=int)
    parser.add_argument('--L_token', default=48, type=int)
    parser.add_argument('--L_out', default=30, type=int)
    parser.add_argument('--valid_start', default=360, type=int)
    parser.add_argument('--test_start', default=510, type=int)

    # model params
    parser.add_argument('--d_model', default=64, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--e_layers', default=1, type=int)
    parser.add_argument('--d_layers', default=1, type=int)
    parser.add_argument('--d_ff', default=64 * 4, type=int)
    parser.add_argument('--factor', default=5, type=int)
    parser.add_argument(
        '--distil',
        action='store_false',
        help='whether to use distilling in encoder, using this argument means not using distilling',
        default=True,
    )
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument(
        '--output_attention', action='store_true', help='whether to output attention in ecoder', default=False
    )
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--dropout', default=0.00, type=float)

    args = parser.parse_args()

    main(args)
