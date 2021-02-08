import numpy as np
import trimesh
from pykdtree.kdtree import KDTree
from data_processing.implicit_waterproofing import implicit_waterproofing

# taken from ifnet: https://github.com/jchibane/if-net/blob/master/data_processing/evaluation.py
## mostly apdopted from occupancy_networks/im2mesh/common.py and occupancy_networks/im2mesh/eval.py

def eval_mesh( mesh_pred, mesh_gt, bb_min, bb_max, n_points=100000):

    pointcloud_pred, idx = mesh_pred.sample(n_points, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_gt, idx = mesh_gt.sample(n_points, return_index=True)
    pointcloud_gt = pointcloud_gt.astype(np.float32)
    normals_gt = mesh_gt.face_normals[idx]

    out_dict = eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)


    bb_len = bb_max - bb_min
    bb_samples = np.random.rand(n_points*10, 3) * bb_len + bb_min

    occ_pred = implicit_waterproofing(mesh_pred, bb_samples)[0]
    occ_gt = implicit_waterproofing(mesh_gt, bb_samples)[0]

    area_union = (occ_pred | occ_gt).astype(np.float32).sum()
    area_intersect = (occ_pred & occ_gt).astype(np.float32).sum()

    out_dict['iou'] =  (area_intersect / area_union)

    return out_dict


def eval_pointcloud(pointcloud_pred, pointcloud_gt,
                    normals_pred=None, normals_gt=None):

    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        normals_gt, normals_pred
    )
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()


    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()


    # Chamfer distance
    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2

    if not normals_pred is None:
        accuracy_normals = accuracy_normals.mean()
        completeness_normals = completeness_normals.mean()
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    else:
        accuracy_normals = np.nan
        completeness_normals = np.nan
        normals_correctness = np.nan


    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
        'iou': np.nan
    }

    return out_dict


def distance_p2p(pointcloud_pred, pointcloud_gt,
                    normals_pred, normals_gt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(pointcloud_gt)
    dist, idx = kdtree.query(pointcloud_pred)

    if normals_pred is None:
        return dist, None

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    normals_dot_product = (normals_gt[idx] * normals_pred).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)

    return dist, normals_dot_product

if __name__ == "__main__":
    from pathlib import Path
    import glob
    #prepare mesh_pathes to read and evaluate
    results_pth = Path("results")
    gt_pth = Path("/media/alex/01D6C1999581FF10/Users/alexs/OneDrive/Desktop/3dfront_share/processed")
    paths_predicted_big_e2e = sorted(glob.glob(str(results_pth/ "exp_endtoend" / "val_**_predicted.obj")))
    paths_predicted_big_skipunet = sorted(glob.glob(str(results_pth/ "exp_skipunet" / "val_**_predicted.obj")))
    paths_predicted_best = sorted(glob.glob(str(results_pth/ "best_pretrained_big" / "val_**_predicted.obj")))
    
    val_split_big = Path("data") / "splits" / "full_data"/ "test.txt"
    val_split_mini2 = Path("data") / "splits" / "mini"/ "test.txt"
    val_split_mini = Path("data") / "splits" / "mini"/ "test_2scale.txt"
            
    with open(str(val_split_big), 'r') as file:
        paths_big_gt = sorted(file.read().splitlines())
        for i, line in enumerate(paths_big_gt):
            paths_big_gt[i] = line + "/mesh.obj"

    with open(str(val_split_mini), 'r') as file:
        paths_mini_gt = file.read().splitlines()
        for i, line in enumerate(paths_mini_gt):
            paths_mini_gt[i] = line + "/mesh.obj"

    with open(str(Path("results")/"mini_results_ordered.txt"),'r') as file:
        paths_predicted_mini = file.read().splitlines()
    
    #print(paths_predicted_mini[:11], paths_mini_gt[:11])
    #evaluation here
       
    #define dict
    performance = {'completeness': [], 'accuracy': [],'normals completeness': [],'normals accuracy': [], 'normals': [], 'completeness2': [], 'accuracy2': [], 'chamfer_l2': [], 'iou': []}
    
    #define variables // argparse this & paths & size
    experiment_name = "exp_mini.txt"
    experiment_results = paths_predicted_mini
    gt_path = paths_mini_gt
    verbose = True
    size = np.array([139,104,112])
    size = np.round(size / 2)
    #print(paths_mini_gt, paths_predicted_mini)
    if verbose:
        print(size)
    
    #evaluate meshes
    for i in range(len(experiment_results)):
        #read meshes
        if verbose:
            print('reading mesh: '+str(i)+'/'+str(len(experiment_results)))

        pred_mesh = trimesh.load(str(experiment_results[i]))
        gt_mesh = trimesh.load(str(gt_path[i]))
        out_dict = eval_mesh(pred_mesh, gt_mesh, 0, size, n_points=100000)

        for key in performance.keys():
            performance[key].append(out_dict[key])
    
    #write file
    with open(str(results_pth/experiment_name), 'w') as file:
        n = len(performance['completeness'])
        file.write(str(n)+' meshes'+'\n')
        for key in performance.keys():
            file.write('mean '+key+': '+str(np.sum(performance[key])/n)+'\n') #avg ('completeness'): ...
        file.write('\n')
        for key in performance.keys():
            file.writelines(key+": "+str(performance[key]))
            file.write('\n')
