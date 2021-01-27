import torch
import torch.nn.functional as F

import numpy as np
import pyexr
from util.visualize import visualize_point_list, visualize_grid
from data_processing.distance_to_depth import depth_to_gridspace

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

def pc_voxels(points, dims, eps=1e-6):
    "Create voxels of `[size]*3` from pointcloud `pc`"

    vox_size = torch.tensor(dims, device=device) #voxel size
    bs = points.size(0) #batch size
    n = points.size(1) #number of points
    #bs = 1
    #n = points.shape[0]

    # check borders
    valid = torch.all((points < 0.5  - eps) & (points > -0.5 + eps), axis=-1).view(-1)
    
    grid = (points + 0.5) * (vox_size - 1)
    #grid = torch.unsqueeze(grid, 0) #if points are unbatched, add fake dimension for batches (testing purposes)
    grid_floor = grid.floor()
    
    grid_idxs = grid_floor.long()
    batch_idxs = torch.arange(bs)[:, None, None].repeat(1, n, 1).to(points.device)

    #print(bs, n, batch_idxs.shape, grid_idxs.shape)

    # idxs of form [batch, z, y, x] where z, y, x discretized indecies in voxel
    idxs = torch.cat([batch_idxs, grid_idxs], dim=-1).view(-1, 4)
    idxs = idxs[valid]

    # trilinear interpolation
    r = grid - grid_floor
    rr = [1. - r, r]
    
    voxels = []
    voxels_t = points.new(bs, dims[0], dims[1], dims[2]).fill_(0)

    def trilinear_interp(pos):
        update = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        update = update.view(-1)[valid]
        
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

def smoothing_kernel(sigma, kernel_size=21):
    #"Generate 3 separate gaussian kernels with `sigma` stddev"
    x = torch.arange(-kernel_size//2 + 1., kernel_size//2 + 1., device=device)
    kernel_1d = torch.exp(-x**2 / (2. * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    k1 = kernel_1d.view(1, 1, 1, 1, -1)
    k2 = kernel_1d.view(1, 1, 1, -1, 1)
    k3 = kernel_1d.view(1, 1, -1, 1, 1)
    return [k1, k2, k3]

def voxels_smooth(voxels, kernels, scale=None):
    #"Apply gaussian blur to voxels with separable `kernels` then `scale`"
    assert isinstance(kernels, list)

    # add fake channel for convs
    bs = voxels.size(0)
    voxels = voxels.unsqueeze(0)

    for k in kernels:
        # add padding for kernel dimension
        padding = [0] * 3
        padding[np.argmax(k.shape) - 2] = max(k.shape) // 2
        #k = k.double()
        #voxels = voxels.double()
        voxels = F.conv3d(voxels, k.repeat(bs, 1, 1, 1, 1), stride=1, padding=padding, groups=bs)

    voxels = voxels.squeeze(0)

    if scale is not None:
        voxels = voxels * scale.view(-1, 1, 1, 1)
        voxels = voxels.clamp(0, 1)

    return voxels

def voxel_occ_from_depth(depth_map, sigma=0.01, kernel_size=3):
    intrinsic_path = (Path("data") / "raw" / "overfit" / "00000" / "intrinsic.txt")
    depth_grid_space = depth_to_gridspace(distance_map_path, intrinsic_path)
    depth_grid_space = norm_grid_space(depth_grid_space)
    voxelized_occupancy = pc_voxels(depth_grid_space, dims)
    smoothed_voxelized_occupancy = voxels_smooth(voxelized_occupancy, kernels=smoothing_kernel(sigma, kernel_size)).unsqueeze(1)
    return smoothed_voxelized_occupancy

def voxel_occ_from_pc(point_cloud, sigma=0.01, kernel_size=3):
    dims=(139, 104, 112)
    point_cloud = norm_grid_space(point_cloud)
    voxelized_occupancy = pc_voxels(point_cloud, dims)
    smoothed_voxelized_occupancy = voxels_smooth(voxelized_occupancy, kernels=smoothing_kernel(sigma, kernel_size)).unsqueeze(1)
    return smoothed_voxelized_occupancy

def norm_grid_space(point_cloud, dims=(139, 104, 112)):
    # center
    #print(point_cloud.shape)
    point_cloud[:,:, 0] -= (dims[0] / 2)
    point_cloud[:,:, 1] -= (dims[1] / 2)
    point_cloud[:,:, 2] -= (dims[2] / 2)

    # scale
    #print(point_cloud.shape)
    point_cloud[:,:, 0] /= dims[0]
    point_cloud[:,:, 1] /= dims[1]
    point_cloud[:,:, 2] /= dims[2]
    #dim_scaling = torch.tensor(dims)
    #point_cloud /= dim_scaling #values between -0.5 & 0.5

    return point_cloud


if __name__ == "__main__":
    from pathlib import Path

    distance_map_path = str(Path("data") / "raw" / "overfit" / "00000" / "distance.exr")
    intrinsic_path = (Path("data") / "raw" / "overfit" / "00000" / "intrinsic.txt")
    output_pt_cloud_path = Path("data") / "visualizations" / "overfit" / "00000" / "diff_depth.obj"
    
    depth_grid_space = depth_to_gridspace(distance_map_path, intrinsic_path) #in voxel values s (36 to 76)*0.05cm
    visualize_point_list(depth_grid_space, output_pt_cloud_path)
    dims = (139, 104, 112)

    # visualize as voxels
    output_voxel_path = Path("data") / "visualizations" / "overfit" / "00000" / "diff_depth_voxels.obj"
    
    #center & scale
    depth_grid_space = norm_grid_space(depth_grid_space, dims)

    #differential voxelization
    voxels = pc_voxels(depth_grid_space, dims)
    smooth = voxels_smooth(voxels, kernels=smoothing_kernel(0.01, 3)).squeeze(0)

    #nondifferential voxelization
    #grid = np.zeros(dims)
    #to_int = lambda x: np.round(x).astype(np.int32)
    #grid[to_int(depth_grid_space[:, 0]), to_int(depth_grid_space[:, 1]), to_int(depth_grid_space[:, 2])] = 1
    
    visualize_grid(smooth, output_voxel_path)

    # lets also visualize in occupancy space, which is just normalized grid space
    output_pt_cloud_path = Path("data") / "visualizations" / "overfit" / "00000" / "diff_depth_occupied.obj"
    visualize_point_list(depth_grid_space, output_pt_cloud_path)

