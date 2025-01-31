import pytorch_lightning as pl
import torch
from pathlib import Path

from model.ifnet import IFNet, implicit_to_mesh
from util import arguments
from util.visualize import visualize_sdf
from dataset.implicit_dataset import ImplicitDataset

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import numpy as np


class ImplicitRefinementTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(ImplicitRefinementTrainer, self).__init__()
        self.hparams = kwargs
        self.ifnet = IFNet()

        self.dims = np.array((139, 104, 112), dtype=np.float32)
        self.dims = np.round(self.dims / self.hparams.scale_factor)

        self.dataset = lambda split: ImplicitDataset(split, self.hparams.datasetdir, self.hparams.num_points, self.hparams.splitsdir)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.ifnet.parameters(), lr=self.hparams.lr)
        return [opt_g], []

    def train_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def forward(self, batch):
        logits = self.ifnet(batch['input'], batch['points'])
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['occupancies'], reduction='none').sum(-1).mean()
        return {'loss': ce_loss}

    def validation_step(self, batch, batch_idx):
        output_vis_path = Path("runs") / self.hparams.experiment / f"vis" / f'{(self.global_step // 1000):05d}'
        output_vis_path.mkdir(exist_ok=True, parents=True)
        for item_idx in range(len(batch['name'])):
            base_name = batch["name"][0]
            implicit_to_mesh(self.ifnet, batch['input'], np.round(self.dims).astype(np.int32), 0.5, output_vis_path / f"{base_name}_predicted.obj")
            visualize_sdf(batch['target'].squeeze().cpu().numpy(), output_vis_path / f"{base_name}_gt.obj", level=1)
        return {'loss': 0}


def train_implicit_refinement(args):
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join("runs", args.experiment, 'checkpoints'), save_top_k=-1, verbose=False, period=args.save_epoch)
    model = ImplicitRefinementTrainer(args)
    trainer = Trainer(gpus=[args.gpu], num_sanity_val_steps=args.sanity_steps, checkpoint_callback=checkpoint_callback, max_epochs=args.max_epoch, limit_val_batches=args.val_check_percent,
                      val_check_interval=min(args.val_check_interval, 1.0), check_val_every_n_epoch=max(1, args.val_check_interval), resume_from_checkpoint=args.resume, logger=None, benchmark=True)

    trainer.fit(model)


if __name__ == '__main__':
    _args = arguments.parse_arguments()
    train_implicit_refinement(_args)
