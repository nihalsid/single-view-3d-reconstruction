import pytorch_lightning as pl
import torch
from torch.nn.functional import interpolate 
from pathlib import Path

from model.ifnet import IFNet, implicit_to_mesh
from model.unet import Unet, UNetMini

from util import arguments
from util.visualize import visualize_sdf, visualize_point_list, visualize_depthmap, visualize_grid
from dataset.scene_net_data import scene_net_data

from model.projection import project
from data_processing.mesh_occupancies import determine_occupancy

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
        self.kernel_size = self.hparams.kernel_size

        self.dims = torch.tensor([139, 104, 112], device=self.device)
        self.dims = (self.dims / self.hparams.scale_factor).round().long()

        self.project = project(self.dims, self.kernel_size, torch.tensor(self.hparams.sigma))
        if self.hparams.resize_input:
            self.unet = Unet(channels_in=3, channels_out=1)
        else:
            self.unet = UNetMini(channels_in=3, channels_out=1)

        if self.hparams.skip_unet:
            self.unet = None

        self.dataset = lambda split: scene_net_data(split, self.hparams.datasetdir, self.hparams.num_points, self.hparams.splitsdir, self.hparams)

    #Here you could set different learning rates for different layers
    def configure_optimizers(self):
        opt_g = torch.optim.Adam([
            {'params': self.unet.parameters(), 'lr':self.hparams.lr},
            {'params': self.project.parameters(), 'lr':10*self.hparams.lr},
            {'params': self.ifnet.parameters(), }
            ], lr=self.hparams.lr)
        if self.hparams.skip_unet:
            opt_g = torch.optim.Adam([            
            {'params': self.project.parameters(), 'lr':10*self.hparams.lr}, {'params': self.ifnet.parameters(), } 
            ], lr=self.hparams.lr)
        return [opt_g], []
    
    def train_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def test_dataloader(self):
        dataset = self.dataset('test')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
    
    def forward(self, batch):
        if not self.hparams.skip_unet:
            depthmap_original = self.unet(batch['rgb'])
            # Resize back and remove extra zero padding if input was resized
            if self.hparams.resize_input:
                resized_depthmap = interpolate(depthmap_original, size= 320, mode='bilinear')
                logits = resized_depthmap[:, :, 40:280, :].squeeze(1) #TODO: Check whether the removed padding affects the gradient in any way
            else: 
                logits = depthmap_original
            
            # Apply sigmoid and renormalisation the values so the predicted depths fall within the per-dataset min and max values.
            renormalized_depthmap = torch.sigmoid(logits) * (self.hparams.max_z - self.hparams.min_z) + self.hparams.min_z
        else:
            renormalized_depthmap = batch['depthmap_target']

        # Forward outputs, ifnet wants points in normed gridspace (-0.5, 0.5)
        point_cloud = self.project.depthmap_to_gridspace(renormalized_depthmap, self.hparams.scale_factor)
        point_cloud = self.project.norm_grid_space(point_cloud)
        # Diff voxelized occupancy from point_cloud -> ifnet -> logits
        voxel_occupancy = self.project(point_cloud)

        #use subset of projected pointcloud
        if self.hparams.subsample_points < (240*320) & self.hparams.subsample_points > 0: 
            indices = torch.randperm(len(point_cloud[0])).to(dtype=torch.long)
            indices = indices[:self.hparams.subsample_points]
            point_cloud = point_cloud[:,indices,:].contiguous() #select n random points per batch
            points = torch.cat((point_cloud, batch['points']), axis=1)
        elif self.hparams.subsample_points == 0:
            points = batch['points']
        else:
            points = torch.cat((point_cloud, batch['points']), axis=1)

        logits_depth = self.ifnet(voxel_occupancy, points)
        
        return logits_depth, renormalized_depthmap, point_cloud

    def training_step(self, batch, batch_idx):
        #forward with additional training supervision
        logits, depthmap, point_cloud = self.forward(batch)
        # additional supervision
        if self.hparams.subsample_points == 0:
            occupancies = batch['occupancies']
        else:
            _, occupancies_pointcloud = determine_occupancy(batch['mesh'], point_cloud.cpu().detach().numpy())
            occupancies_pointcloud = occupancies_pointcloud.to(logits.device)     
            occupancies = torch.cat((occupancies_pointcloud, batch['occupancies']), axis=1)
        
        #losses and logging
        loss = self.losses_and_logging(batch, depthmap, logits, occupancies, 'train')
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        logits, depthmap, point_cloud = self.forward(batch)
        
        # additional supervision
        if self.hparams.subsample_points == 0:
            occupancies = batch['occupancies']
        else:
            _, occupancies_pointcloud = determine_occupancy(batch['mesh'], point_cloud.cpu().detach().numpy())
            occupancies_pointcloud = occupancies_pointcloud.to(logits.device)     
            occupancies = torch.cat((occupancies_pointcloud, batch['occupancies']), axis=1)

        if self.hparams.visualize:
            self.visualize_intermediates(batch, depthmap, point_cloud)

        #losses and logging
        loss = self.losses_and_logging(batch, depthmap, logits, occupancies, 'val')
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        _, depthmap, point_cloud = self.forward(batch)
        self.visualize_intermediates(batch, depthmap, point_cloud)

        return {'loss': 0}
    
    def losses_and_logging(self, batch, depthmap, logits, occupancies, mode):

        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, occupancies, reduction='mean')
        mse_loss = torch.nn.functional.mse_loss(depthmap, batch['depthmap_target'], reduction='mean')
        loss = ce_loss + mse_loss

        mesh_ce_loss = ce_loss
        if self.hparams.subsample_points > 0:
            logits_mesh, occ_mesh = logits[:,self.hparams.subsample_points:].contiguous(), occupancies[:,self.hparams.subsample_points:].contiguous()
            mesh_ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_mesh, occ_mesh, reduction='mean')     
        
        #grid in (z, y, x) orientation where z = viewing direction
        self.log('sigma_x', self.project.sigma[2])
        self.log('sigma_y', self.project.sigma[1])
        self.log('sigma_z', self.project.sigma[0])
        self.log(f'{mode}_mesh_ce_loss', mesh_ce_loss)
        self.log(f'{mode}_ce_loss', ce_loss)
        self.log(f'{mode}_mse_depth_loss', mse_loss)
        self.log(f'{mode}_loss', loss)

        if self.hparams.no_depth_sup:
            return ce_loss

        return loss

    def visualize_intermediates(self, batch, depthmap, point_cloud):
        #prepare items
        voxel_occupancy = self.project(point_cloud)
        #bring point_cloud back from normed space to mesh space for visualization
        point_cloud = self.project.un_norm_grid_space(point_cloud)
        
        for i in range(len(batch['name'])):
            #prepare item names
            output_vis_path = Path("runs") / self.hparams.experiment / f"vis" / f'{(self.global_step // 100):05d}'
            output_vis_path.mkdir(exist_ok=True, parents=True)
            base_name = "_".join(batch["name"][i].split("/")[-3:])   
            
            #visualize outputs of network stages (depthmap, voxelgrid, pointcloud, mesh)
            #visualize_point_list(point_cloud[i].squeeze(), output_vis_path / f"{base_name}_pc.obj")
            visualize_grid(voxel_occupancy[i].squeeze().cpu().numpy(), output_vis_path / f"{base_name}_voxelized.obj")
            implicit_to_mesh(self.ifnet, voxel_occupancy[i].unsqueeze(0), self.dims.cpu().detach().numpy().astype(np.int32), 0.5, output_vis_path / f"{base_name}_predicted.obj", res_increase=self.hparams.inf_res)
            
            #visualize_sdf(batch['target'][i].squeeze().cpu().numpy(), output_vis_path / f"{base_name}_gt.obj", level=1)
            visualize_depthmap(depthmap[i], output_vis_path / f"{base_name}_depthmap", flip = True)

    
    #uncomment to log gradients
    """
    def on_after_backward(self):
    # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            print(self.project.sigma.grad)
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(tag=name, values=grads, global_step=self.trainer.global_step)"""


def use_pretrained_unet(args):
    model = SceneNetTrainer(args)
    pretrained_dict = torch.load(args.pretrain_unet)['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys & leave only unet keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('unet' in k)} 
    # 2. overwrite entries in the existing state dict
    model.load_state_dict(pretrained_dict, strict=False)
    return model    


def train_scene_net(args):
    seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join("runs", args.experiment, 'checkpoints'), save_top_k=2, save_last=True, monitor='val_ce_loss', verbose=False, period=args.save_epoch)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join("runs", 'logs/'), name=args.experiment)
    if args.resume is None and args.pretrain_unet is None:
        model = SceneNetTrainer(args)
    elif args.resume is not None:
        model = SceneNetTrainer(args).load_from_checkpoint(args.resume)
    elif args.pretrain_unet is not None:
        model = use_pretrained_unet(args)
    
    trainer = Trainer(
        gpus=[args.gpu] , num_sanity_val_steps=args.sanity_steps, checkpoint_callback=checkpoint_callback, max_epochs=args.max_epoch, 
        limit_val_batches=args.val_check_percent, val_check_interval=min(args.val_check_interval, 0.5), 
        check_val_every_n_epoch=max(1, args.val_check_interval), resume_from_checkpoint=args.resume, logger=tb_logger, benchmark=True, 
        profiler=args.profiler, precision=args.precision
        )

    ## for testing specific models
    if args.test is not None:
        model = SceneNetTrainer.load_from_checkpoint(args.test)
        #change specific model-hparams for testing
        model.hparams.inf_res = args.inf_res #default 1
        model.hparams.scale_factor = args.scale_factor #default 1
        model.hparams.skip_unet = args.skip_unet #default False
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    _args = arguments.parse_arguments()
    train_scene_net(_args)
