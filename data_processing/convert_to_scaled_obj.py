import trimesh
import numpy as np
from pathlib import Path
from util.arguments import argparse
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert meshes to scaled and centered meshes'
    )

    parser.add_argument('--experiment', type=str, default="asd")
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose')
    parser.add_argument('--scale_factor' ,type=int, default=1, help='Down scale the voxel grid input.')
    args = parser.parse_args()

    dims = np.array([139, 104, 112])
    dims = dims / np.round(args.scale_factor).astype(np.long)

    results_path = Path('results')
    source_path = args.experiment
    
    source_path = results_path / source_path
    meshes = glob.glob(str(source_path / '*_predicted.obj'))

    #for mini only
    #meshes = glob.glob(str(source_path / '0491'/'*'/ 'mesh.obj'))
    #meshes.extend(glob.glob(str(source_path / '0492'/'*'/ 'mesh.obj')))
    
    #print(scenes_folders)
    my_names = []
    for i in range(len(meshes)):
        #read meshes
        if args.verbose:
            print('reading mesh: '+str(i)+'/'+str(len(meshes)))
        
        mesh = trimesh.load(str(meshes[i]))
        mesh.apply_translation(-dims / 2)
        mesh.apply_scale(1 / dims)

        name = meshes[i].split('/')[-1]

        #for normalizing from data folder
        #name = '_'.join(meshes[i].split('/')[-3:])
                
        mesh.export(str(source_path / name[:-4]) + '_normed.obj')
        #my_names.append(str(Path('results/normed_gt_val') / name[:-4]) + '_normed.obj')

    """with open('normed_big_gt.txt', 'w') as f:
        f.write('\n'.join(my_names))"""

# folder structure:
# if-net_scenes:
# -- results
# ---- experiment
# ------ meshes with the names val_{scene}_ {view}_predicted.obj