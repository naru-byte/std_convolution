import argparse
import random
from os.path import join, isfile

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from src.data.sssd2019_plmodule import SSSD2019DataModule
from src.models.sssd2019_3dcnn import SSSD2019Cnn, SSSD2019DYCnn, SSSD2019ShapeDYCnn

val_speakers = ["s007","s009","s015","s021","s023","s029",
                "s031","s036","s040","s044","s049","s053",
                "s056","s060","s066","s070","s072","s075"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/datasets/SSSD2019")
    parser.add_argument('--model', default="cnn")
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/sssd2019')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frontend_dims", type=str, default='32:32:32:64:64:64:64:64:32')
    parser.add_argument("--pooling_layer", type=str, default='3:6:9')
    parser.add_argument("--dynamic_layer", type=str, default='1')
    parser.add_argument("--num_weights", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--max_timesteps", type=int, default=64)

    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--epochs_num", type=int, default=50)
    parser.add_argument("--non_progressbar", default=False, action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    progress = 1 if not args.non_progressbar else 0
    assert args.checkpoint != None
    assert isfile(args.checkpoint)

    dm = SSSD2019DataModule(hparams=args, val_speakers=val_speakers)
    dm.setup()

    if args.model == 'cnn':
        model = SSSD2019Cnn(
            hparams=args,
            in_channels=args.in_channels,
            num_classes=dm.num_classes
        )
    elif args.model == "dycnn":
        model = SSSD2019DYCnn(
            hparams=args,
            in_channels=args.in_channels,
            num_classes=dm.num_classes,
            epochs_num=args.epochs_num
        )
    elif args.model == "s_dycnn":
        model = SSSD2019ShapeDYCnn(
            hparams=args,
            in_channels=args.in_channels,
            num_classes=dm.num_classes,
            epochs_num=args.epochs_num
        )
    else:
        assert args.model == 'cnn'

    logger = WandbLogger(project="sssd2019", name=args.wandb_name)
    model.logger = logger

    ckpt = torch.load(args.checkpoint)
    model.on_load_checkpoint(ckpt)
    if args.checkpoint.split("/")[-1] == 'last_state.ckpt':
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt['state_dict'])

    dm.setup("test")

    trainer = Trainer(gpus=1,
                      progress_bar_refresh_rate=progress)
    trainer.test(model, dm.test_dataloader())