import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#smear points into 3d Gaussians with mean
#slow:
#evaluate the points normed gaussian p(x,y,z) at every voxel center (x',y',z')
#clamp values between 0,1
#feed these voxelized values into ifnet

#fast:
#interpolate values to values of nearest gridpoints
#convolve over gridpoints with gaussians
#implementation heavily inspired by github.com/puhsu/point_clouds
#https://arxiv.org/abs/1810.09381

class project(nn.Module):
    #Module: Projection from Depthmap to Pointcloud & Differentiable voxelization of point cloud
    def __init__(self, dims, kernel_size=3, sigma=0.01):
        super(project, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = torch.nn.Parameter(sigma)
        self.sigma.requires_grad = True
        self.vox_size = torch.tensor(dims, device = device)
        self.intrinsic = self.get_intrinsic()
        # for testing purposes
        #self.intrinsic = torch.nn.Parameter(self.intrinsic)
        #self.intrinsic.requires_grad = True

    def forward(self, point_cloud):
        #Voxelize pointcloud
        voxel_occupancy = self.voxel_occ_from_pc(point_cloud)
        return voxel_occupancy

    def pc_voxels(self, points, eps=1e-6):
        bs = points.size(0) #batch size
        n = points.size(1) #number of points

        # check borders
        valid = torch.all((points < 0.5  - eps) & (points > -0.5 + eps), axis=-1).view(-1)
        
        grid = (points + 0.5) * (self.vox_size - 1)
        grid_floor = grid.floor()
         
        grid_idxs = grid_floor.long()
        batch_idxs = torch.arange(bs, device=device)[:, None, None].repeat(1, n, 1)

        # idxs of form [batch, z, y, x] where z, y, x discretized indecies in voxel
        idxs = torch.cat([batch_idxs, grid_idxs], dim=-1).view(-1, 4)
        idxs = idxs[valid]

        # trilinear interpolation
        r = grid - grid_floor
        rr = [1. - r, r]
        
        voxels = []
        voxels_t = points.new(bs, self.vox_size[0], self.vox_size[1], self.vox_size[2]).fill_(0)

        def trilinear_interp(pos):
            update = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
            update = update.view(-1)[valid]
            
            #ideally construct on device (not .to(device)) but not sure how
            shift_idxs = torch.LongTensor([[0] + pos]).to(points.device)
            shift_idxs = shift_idxs.repeat(idxs.size(0), 1)
            update_idxs = idxs + shift_idxs
            
            #valid_shift = update_idxs < size
            voxels_t.index_put_(torch.unbind(update_idxs, dim=1), update, accumulate=True)
            return voxels_t
            
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    voxels.append(trilinear_interp([k, j, i]))

        return torch.stack(voxels).sum(dim=0).clamp(0, 1)

    def smoothing_kernel(self):
        #"Generate 3 separate gaussian kernels with `sigma` stddev"
        x = torch.arange(-self.kernel_size//2 + 1., self.kernel_size//2 + 1., device=device)
        kernel_1d = torch.exp(-x**2 / (2. * self.sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        k1 = kernel_1d.view(1, 1, 1, 1, -1)
        k2 = kernel_1d.view(1, 1, 1, -1, 1)
        k3 = kernel_1d.view(1, 1, -1, 1, 1)
        
        return [k1, k2, k3]

    def voxels_smooth(self, voxels, kernels):
        #"Apply gaussian blur to voxels with separable `kernels` then `scale`"
        assert isinstance(kernels, list)

        # add fake channel for convs
        bs = voxels.size(0)
        voxels = voxels.unsqueeze(0)

        for k in kernels:
            # add padding for kernel dimension
            padding = [0] * 3
            padding[np.argmax(k.shape) - 2] = max(k.shape) // 2
            voxels = F.conv3d(voxels, k.repeat(bs, 1, 1, 1, 1), stride=1, padding=padding, groups=bs)

        voxels = voxels.squeeze(0).clamp(0,1)
        return voxels

    def voxel_occ_from_pc(self, point_cloud):
        voxelized_occupancy = self.pc_voxels(point_cloud)
        smoothed_voxelized_occupancy = self.voxels_smooth(voxelized_occupancy, kernels=self.smoothing_kernel())
        return smoothed_voxelized_occupancy.unsqueeze(1)

    def norm_grid_space(self, pc):
        # center & scale point_cloud values between -0.5 & 0.5#
        pc[:,:, 0] = pc[:,:, 0] - (self.vox_size[0] / 2)
        pc[:,:, 1] = pc[:,:, 1] - (self.vox_size[1] / 2)
        pc[:,:, 2] = pc[:,:, 2] - (self.vox_size[2] / 2)
        pc[:,:, 0] = pc[:,:, 0] / self.vox_size[0]
        pc[:,:, 1] = pc[:,:, 1] / self.vox_size[1]
        pc[:,:, 2] = pc[:,:, 2] / self.vox_size[2]
        return pc

    def un_norm_grid_space(self, point_cloud):
        #inplace operations (pc[:,0] += 1) do not create copys but access the tensor in memory directly. Loses gradients
        #use pc[:,0] = pc[:,0] + 1 instead to create a temporary copy & allow gradient prop. 
        # See https://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html and https://discuss.pytorch.org/t/what-is-in-place-operation/16244/15
        
        # Bring values back to gridspace ([0,0,0] - dims)
        point_cloud[:,:, 0] = point_cloud[:,:, 0] * self.vox_size[0]
        point_cloud[:,:, 1] = point_cloud[:,:, 1] * self.vox_size[1]
        point_cloud[:,:, 2] = point_cloud[:,:, 2] * self.vox_size[2]

        point_cloud[:,:, 0] = point_cloud[:,:, 0] + (self.vox_size[0] / 2)
        point_cloud[:,:, 1] = point_cloud[:,:, 1] + (self.vox_size[1] / 2)
        point_cloud[:,:, 2] = point_cloud[:,:, 2] + (self.vox_size[2] / 2)

        return point_cloud

    def depthmap_to_gridspace(self, depthmap, scale_factor=1):
        focal_length, cx, cy = self.intrinsic[0][0], self.intrinsic[0][2], self.intrinsic[1][2]
        bs = depthmap.shape[0] #batch_size

        X, Y, Z = self.depth_to_camera(depthmap, focal_length, cx, cy)
        self.intrinsic_inv = torch.inverse(self.intrinsic)
        frustum = self.generate_frustum([320, 240], self.intrinsic_inv, 0.4, 6.0)
        dims, camera2frustum = self.generate_frustum_volume(frustum, 0.05 * scale_factor)

        # depth from camera to grid space
        coords = torch.stack([X, Y, Z, torch.ones_like(X)])
        depth_in_gridspace = (camera2frustum @ coords)[:3, :].transpose(1,0).reshape(bs, -1, 3)

        return depth_in_gridspace

    @staticmethod
    def generate_frustum(image_size, intrinsic_inv, depth_min, depth_max):
        x = image_size[0]
        y = image_size[1]
        eight_points = torch.tensor([[0 * depth_min, 0 * depth_min, depth_min, 1.0],
                                [0 * depth_min, y * depth_min, depth_min, 1.0],
                                [x * depth_min, y * depth_min, depth_min, 1.0],
                                [x * depth_min, 0 * depth_min, depth_min, 1.0],
                                [0 * depth_max, 0 * depth_max, depth_max, 1.0],
                                [0 * depth_max, y * depth_max, depth_max, 1.0],
                                [x * depth_max, y * depth_max, depth_max, 1.0],
                                [x * depth_max, 0 * depth_max, depth_max, 1.0]], device=intrinsic_inv.device).transpose(1, 0)
        frustum = (torch.mm(intrinsic_inv, eight_points)).transpose(1, 0)
        
        return frustum[:, :3]

    @staticmethod
    def generate_frustum_volume(frustum, voxelsize):
        maxx = torch.max(frustum[:, 0]) / voxelsize
        maxy = torch.max(frustum[:, 1]) / voxelsize
        maxz = torch.max(frustum[:, 2]) / voxelsize
        minx = torch.min(frustum[:, 0]) / voxelsize
        miny = torch.min(frustum[:, 1]) / voxelsize
        minz = torch.min(frustum[:, 2]) / voxelsize

        dimX = torch.ceil(maxx - minx)
        dimY = torch.ceil(maxy - miny)
        dimZ = torch.ceil(maxz - minz)
        camera2frustum = torch.tensor([[1.0 / voxelsize, 0, 0, -minx],
                                [0, 1.0 / voxelsize, 0, -miny],
                                [0, 0, 1.0 / voxelsize, -minz],
                                [0, 0, 0, 1.0]], device=frustum.device)

        return (dimX, dimY, dimZ), camera2frustum

    @staticmethod
    def depth_to_camera(depth_map, f, cx, cy):
        v, u = torch.meshgrid(torch.arange(depth_map.shape[-2], device=depth_map.device), torch.arange(depth_map.shape[-1], device=depth_map.device))
        X = ((torch.multiply(u, depth_map) - cx * depth_map) / f)
        Y = -((torch.multiply(v, depth_map) - cy * depth_map) / f)
        Z = depth_map
        return X.flatten(), Y.flatten(), Z.flatten()

    @staticmethod
    def get_intrinsic(intrinsic_path=None):
        if intrinsic_path is None:
            intrinsic_path = (Path("data") / "raw" / "overfit" / "00000" / "intrinsic.txt")

        intrinsic_line_0, intrinsic_line_1 = intrinsic_path.read_text().splitlines()[:2]
        focal_length = float(intrinsic_line_0[2:].split(',')[0])
        cx = float(intrinsic_line_0[2:-2].split(',')[2].strip())
        cy = float(intrinsic_line_1[1:-2].split(',')[2].strip())
        intrinsic = torch.tensor([[focal_length, 0, cx, 0], [0, focal_length, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device)
        return intrinsic

if __name__ == '__main__':
    from torchviz import make_dot, make_dot_from_trace
    import pyexr
    import graphviz


    model = project(11, torch.tensor(3.))
    dm_path = Path("runs") /"02020244_fast_dev"/"vis"/"00000"/ "val_0492_19_depthmap.exr"
    depth_map = pyexr.open(str(dm_path)).get("Z")[:, :, 0]
    depth_map = torch.from_numpy(depth_map).to(device)
    model.to(device)
    pointcloud = model.depthmap_to_gridspace(depth_map).reshape(-1,3).unsqueeze(0)
    voxelized_occ = model(pointcloud)
    test = model(model.depthmap_to_gridspace(depth_map).reshape(-1,3).unsqueeze(0))
    a = make_dot(test, params=dict(model.named_parameters()))
    a.render(filename='backwards_intrinsic.png', format='png')
    #visualize_point_list(pointcloud, output_pt_cloud_path)