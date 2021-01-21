import pytorch_lightning as pl
import torch
from torch.nn.functional import interpolate 
from pathlib import Path

from model.ifnet import IFNet, implicit_to_mesh
from model.unet import Unet, UNetMini

from util import arguments
from util.visualize import visualize_sdf
from util.visualize import visualize_point_list
from util.visualize import visualize_depthmap
from dataset.scene_net_data import scene_net_data

from data_processing.pointcloud2voxels3d_fast import voxel_occ_from_pc
from data_processing.distance_to_depth import depthmap_to_gridspace

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import os
import numpy as np


class SceneNetTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(SceneNetTrainer, self).__init__()
        self.hparams = kwargs
        self.ifnet = IFNet()
        if self.hparams.resize_input:
            self.unet = Unet(channels_in=3, channels_out=1)
        else:
            self.unet = UNetMini(channels_in=3, channels_out=1)
        self.dims = np.array((139, 104, 112), dtype=np.float32)
        self.dataset = lambda split: scene_net_data(split, self.hparams.datasetdir, self.hparams.num_points, self.hparams.splitsdir)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(list(self.unet.parameters()) + list(self.ifnet.parameters()), lr=self.hparams.lr)
        return [opt_g], []
    
    def train_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
    
    def forward(self, batch):
        depthmap_original = self.unet(batch['rgb'])

        # Resize back and remove extra zero padding if input was resized
        if self.hparams.resize_input:
            resized_depthmap = interpolate(depthmap, size= 320, mode='bilinear')
            depthmap = resized_depthmap[:, :, 40:280, :] #TODO: Check whether the removed padding affects the gradient in any way
        else: 
            depthmap = depthmap_original
            
        point_cloud = depthmap_to_gridspace(depthmap)
        voxel_occupancy = voxel_occ_from_pc(point_cloud)
        logits_depth = self.ifnet(voxel_occupancy, point_cloud)
        return logits_depth, depthmap

    def training_step(self, batch, batch_idx):
        #forward with additional training supervision
        logits_depth, depthmap = self.forward(batch)
        logits_mesh = self.ifnet(batch['input'], batch['points'])

        #losses
        #reg_loss_depth = torch.nn.functional.mse_loss(depthmap, batch['depthmap_target'], reduction='none').sum(-1).mean()
        ce_loss_mesh = torch.nn.functional.binary_cross_entropy_with_logits(logits_mesh, batch['occupancies'], reduction='none').sum(-1).mean()
        ce_loss_depth = torch.nn.functional.binary_cross_entropy_with_logits(logits_depth, torch.ones_like(logits_depth), reduction='none').sum(-1).mean()
        loss = ce_loss_mesh + 500*ce_loss_depth# + reg_loss_depth
        
        self.log('loss', loss)
        self.log('ce_loss_depth', ce_loss_depth)
        self.log('ce_loss_mesh', ce_loss_mesh)
        #self.log('reg_loss_depth', reg_loss_depth)

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        output_vis_path = Path("runs") / self.hparams.experiment / f"vis" / f'{(self.global_step // 1000):05d}'
        output_vis_path.mkdir(exist_ok=True, parents=True)
        for item_idx in range(len(batch['name'])):

            #prepare items
            base_name = batch["name"][0]
            depthmap = self.unet(batch['rgb'])
            point_cloud = depthmap_to_gridspace(depthmap).squeeze().reshape(-1,3)
            
            #visualize outputs of network stages (depthmap, pointcloud, mesh)
            visualize_depthmap(depthmap, output_vis_path / f"{base_name}_depthmap", flip = True)
            visualize_point_list(point_cloud, output_vis_path / f"{base_name}_pc.obj")
            implicit_to_mesh(self.ifnet, batch['input'], np.round(self.dims).astype(np.int32), 0.5, output_vis_path / f"{base_name}_predicted.obj")
            #visualize_sdf(batch['target'].squeeze().cpu().numpy(), output_vis_path / f"{base_name}_gt.obj", level=1)
        return {'loss': 0}

    def on_after_backward(self):
    # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(tag=name, values=grads, global_step=self.trainer.global_step)
    

def train_scene_net(args):
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join("runs", args.experiment, 'checkpoints'), save_top_k=-1, verbose=False, period=args.save_epoch)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join("runs", 'logs/'))
    if args.resume is None:
        model = SceneNetTrainer(args)
    else:
        model = SceneNetTrainer.load_from_checkpoint(args.resume)
    
    trainer = Trainer(gpus=args.gpu , num_sanity_val_steps=args.sanity_steps, checkpoint_callback=checkpoint_callback, max_epochs=args.max_epoch, limit_val_batches=args.val_check_percent,
                      val_check_interval=min(args.val_check_interval, 1.0), check_val_every_n_epoch=max(1, args.val_check_interval), 
                      resume_from_checkpoint=args.resume, logger=tb_logger, benchmark=True, profiler=args.profiler, precision=args.precision, log_gpu_memory='all')

    trainer.fit(model)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    _args = arguments.parse_arguments()
    train_scene_net(_args)
