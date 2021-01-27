import pytorch_lightning as pl
import torch
from torch.nn.functional import interpolate 
from pathlib import Path

from model.ifnet import IFNet, implicit_to_mesh
from model.unet import Unet, UNetMini

from util import arguments
from util.visualize import visualize_sdf, visualize_point_list, visualize_depthmap, visualize_grid
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
        ####These steps to make sigma & kernelsize of voxelization learnable
        #params = list(net.parameters())
        #params.extend(list(loss.parameters()))
        #opt = torch.optim.Adam(params,lr=1e-3,weight_decay=5e-4)
        #### Per Layer Learning rates
        #optim.SGD([
        #        {'params': model.base.parameters()},
        #        {'params': model.classifier.parameters(), 'lr': 1e-3}
        #    ], lr=1e-2, momentum=0.9)
        self.hparams = kwargs
        self.ifnet = IFNet()
        if self.hparams.resize_input:
            self.unet = Unet(channels_in=3, channels_out=1)
        else:
            self.unet = UNetMini(channels_in=3, channels_out=1)
        self.dims = np.array((139, 104, 112), dtype=np.float32)
        self.dataset = lambda split: scene_net_data(split, self.hparams.datasetdir, self.hparams.num_points, self.hparams.splitsdir, self.hparams)

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
            resized_depthmap = interpolate(depthmap_original, size= 320, mode='bilinear')
            depthmap = resized_depthmap[:, :, 40:280, :].squeeze(1) #TODO: Check whether the removed padding affects the gradient in any way
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
        occupancies_depth = torch.ones_like(logits_depth)

        #losses
        mse_loss = 500*torch.nn.functional.mse_loss(depthmap, batch['depthmap_target'], reduction='mean')
        ce_loss_mesh = torch.nn.functional.binary_cross_entropy_with_logits(logits_mesh, batch['occupancies'], reduction='none').sum(-1).mean()
        ce_loss_depth = torch.nn.functional.binary_cross_entropy_with_logits(logits_depth, occupancies_depth, reduction='none').sum(-1).mean()
        loss = ce_loss_mesh + ce_loss_depth + mse_loss
        
        self.log('loss', loss)
        self.log('ce_loss_depth', ce_loss_depth)
        self.log('ce_loss_mesh', ce_loss_mesh)
        self.log('mse_depth_loss', mse_loss)

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        logits_depth, depthmap = self.forward(batch)
        logits_mesh = self.ifnet(batch['input'], batch['points'])
        occupancies_depth = torch.ones_like(logits_depth)

        #losses
        mse_loss = 500*torch.nn.functional.mse_loss(depthmap, batch['depthmap_target'], reduction='mean')
        ce_loss_mesh = torch.nn.functional.binary_cross_entropy_with_logits(logits_mesh, batch['occupancies'], reduction='none').sum(-1).mean()
        ce_loss_depth = torch.nn.functional.binary_cross_entropy_with_logits(logits_depth, occupancies_depth, reduction='none').sum(-1).mean()
        val_loss = ce_loss_mesh + ce_loss_depth + mse_loss
        
        self.log('val_ce_loss_depth', ce_loss_depth)
        self.log('val_ce_loss_mesh', ce_loss_mesh)
        self.log('val_mse_depth_loss', mse_loss)
        self.log('val_loss', val_loss)

        #only visualize on argument
        if self.hparams.visualize:
            for i in range(len(batch['name'])):
                #prepare item names
                output_vis_path = Path("runs") / self.hparams.experiment / f"vis" / f'{(self.global_step // 100):05d}'
                output_vis_path.mkdir(exist_ok=True, parents=True)
                base_name = "_".join(batch["name"][i].split("/")[-3:])
                
                #prepare items
                point_cloud = depthmap_to_gridspace(depthmap)
                voxel_occupancy = voxel_occ_from_pc(point_cloud)
                #unnormalize pointcloud --> gridspace
                dims = torch.from_numpy(self.dims).to(point_cloud.device)
                point_cloud = (point_cloud[i]*dims).squeeze()
                point_cloud += dims/2
                
                #visualize outputs of network stages (depthmap, pointcloud, mesh)
                visualize_point_list(point_cloud, output_vis_path / f"{base_name}_pc.obj")
                visualize_grid(voxel_occupancy[i].squeeze().cpu().numpy(), output_vis_path / f"{base_name}_voxelized.obj")
                implicit_to_mesh(self.ifnet, voxel_occupancy[i].unsqueeze(0), np.round(self.dims).astype(np.int32), 0.5, output_vis_path / f"{base_name}_predicted.obj")
                visualize_sdf(batch['target'][i].squeeze().cpu().numpy(), output_vis_path / f"{base_name}_gt.obj", level=1)
                visualize_depthmap(depthmap[i], output_vis_path / f"{base_name}_depthmap", flip = True)
        
        return {'val_loss': val_loss}
    
    #uncomment to log gradients
    """
    def on_after_backward(self):
    # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(tag=name, values=grads, global_step=self.trainer.global_step)
    """
    

def train_scene_net(args):
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join("runs", args.experiment, 'checkpoints'), save_top_k=3, monitor='val_loss',verbose=False, period=args.save_epoch)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join("runs", 'logs/'), name='scene_net')
    if args.resume is None and args.pretrain_unet is None:
        model = SceneNetTrainer(args)
    elif args.resume is not None:
        model = SceneNetTrainer(args).load_from_checkpoint(args.resume)
    elif args.pretrain_unet is not None:
        model = SceneNetTrainer(args)
        pretrained_dict = torch.load(args.pretrain_unet)['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model.load_state_dict(pretrained_dict, strict=False)
    
    trainer = Trainer(gpus=args.gpu , num_sanity_val_steps=args.sanity_steps, checkpoint_callback=checkpoint_callback, max_epochs=args.max_epoch, limit_val_batches=args.val_check_percent,
                      val_check_interval=min(args.val_check_interval, 0.5), check_val_every_n_epoch=max(1, args.val_check_interval), 
                      resume_from_checkpoint=args.resume, logger=tb_logger, benchmark=True, profiler=args.profiler, precision=args.precision)#, log_gpu_memory='all')

    trainer.fit(model)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    _args = arguments.parse_arguments()
    train_scene_net(_args)
