import pytorch_lightning as pl
import torch
from pathlib import Path
from PIL import Image

from model.unet import UNetMini
from dataset.scenes_dataset import ScenesDataset
from util import arguments
import pyexr

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms.functional as F
import os
import numpy as np

class DepthRegressorTrainer(pl.LightningModule):
    def __init__(self, kwargs):
        super(DepthRegressorTrainer, self).__init__()
        self.hparams = kwargs
        self.unet = UNetMini(channels_in=3, channels_out=1)
        self.dataset = lambda split: ScenesDataset(split, self.hparams.datasetdir, self.hparams.splitsdir, kwargs)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.unet.parameters(), lr=self.hparams.lr)
        return [opt_g], []

    def train_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def forward(self, batch):
        logits = self.unet(batch['input'])
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        # check the reduction
        mse_loss = torch.nn.functional.mse_loss(logits, batch['target'], reduction='none').sum(-1).mean()
        return {'loss': mse_loss}

    def validation_step(self, batch, batch_idx):
        output_vis_path = Path("runs") / self.hparams.experiment / f"vis" / f'{(self.global_step // 1000):05d}'
        output_vis_path.mkdir(exist_ok=True, parents=True)

        prediction = self.forward(batch)
        
        depth_map = prediction[0].squeeze().cpu().numpy()
        pyexr.write(str(output_vis_path / "depth_map.exr"), depth_map)

        mse_loss = torch.nn.functional.mse_loss(prediction, batch['target'], reduction='none').sum(-1).mean()
        return {'loss': mse_loss}


def train_unet(args):
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join("runs", args.experiment, 'checkpoints'), save_top_k=-1, verbose=False, period=args.save_epoch)
    model = DepthRegressorTrainer(args)
    trainer = Trainer(gpus=[args.gpu], num_sanity_val_steps=args.sanity_steps, checkpoint_callback=checkpoint_callback, max_epochs=args.max_epoch, limit_val_batches=args.val_check_percent,
                      val_check_interval=min(args.val_check_interval, 1.0), check_val_every_n_epoch=max(1, args.val_check_interval), resume_from_checkpoint=args.resume, logger=None, benchmark=True)

    trainer.fit(model)


if __name__ == '__main__':
    _args = arguments.parse_arguments()
    train_unet(_args)
