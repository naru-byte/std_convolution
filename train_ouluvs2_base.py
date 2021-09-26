import argparse
import random
from os.path import join

import psutil
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from src.data.ouluvs2_plmodule import OuluVS2DataModule
from src.models.sssd2019_3dcnn import SSSD2019Cnn, SSSD2019DYCnn, SSSD2019ShapeDYCnn, SSSD2019StdDYCnn
from utils import manually_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/datasets/OuluVS2")
    parser.add_argument('--model', default="cnn")
    parser.add_argument("--checkpoint_dir", type=str, default='data/checkpoints/OuluVS2')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frontend_dims", type=str, default='32:32:32:64:64:64:64')
    parser.add_argument("--pooling_layer", type=str, default='1:3:5')
    parser.add_argument("--dynamic_layer", type=str, default='1')
    parser.add_argument("--num_weights", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--max_timesteps", type=int, default=32)
    parser.add_argument('--data_mode', default="p_d")

    parser.add_argument("--wandb_name", default=None, type=str)
    parser.add_argument("--non_progressbar", default=False, action='store_true')
    parser.add_argument("--non_trainer_training", default=True, action='store_false')
    args = parser.parse_args()

    assert args.data_mode in ['p_d', 'p', 'd']

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    args.workers = psutil.cpu_count(logical=False) if args.workers == None else args.workers
    progress = 1 if not args.non_progressbar else 0

    dm = OuluVS2DataModule(hparams=args)
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
            num_classes=dm.num_classes
        )
    elif args.model == "s_dycnn":
        model = SSSD2019ShapeDYCnn(
            hparams=args,
            in_channels=args.in_channels,
            num_classes=dm.num_classes
        )
    elif args.model == 'ss_dycnn':
        model = SSSD2019StdDYCnn(
            hparams=args,
            in_channels=args.in_channels,
            num_classes=dm.num_classes
        )
    else:
        assert args.model == 'cnn'

    logger = WandbLogger(project="ouluvs2_sdycnn", name=args.wandb_name)
    model.logger = logger

    if args.checkpoint is not None:
        dm.setup("test")

        ckpt = torch.load(args.checkpoint)
        model.on_load_checkpoint(ckpt)
        if args.checkpoint.split("/")[-1] == 'last_state.ckpt':
            model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt['state_dict'])

        trainer = Trainer(gpus=1, progress_bar_refresh_rate=progress)
        trainer.test(model, dm.test_dataloader())

    checkpoint_callback = ModelCheckpoint(
        filepath=join(args.checkpoint_dir,"{epoch}_{val_loss:.2f}_{val_accuracy:.2f}"),
        save_top_k=3,
        monitor="val_loss",
        period=1,
        prefix="ouluvs2"
    )

    trainer = Trainer(
        logger=logger,
        gpus=1,
        max_epochs=args.epochs,
        precision=args.precision,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=progress
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    logger.log_metrics({'parameters':trainable_params})
    logger.log_hyperparams(args)

    if args.non_trainer_training:
        trainer.fit(model, dm)
    else:
        manually_training(model, dm, args.epochs, args.checkpoint_dir)

    torch.save(model.state_dict(), join(args.checkpoint_dir,"last_state.ckpt"))

    trainer.test(model, dm.test_dataloader())