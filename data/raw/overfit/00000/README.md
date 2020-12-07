# Data formats

## distance maps (distance_*.exr):
 - exr, I used: https://github.com/tvogels/pyexr
 - 3 channels, but 3x the same --> just take the first
 - these are distance maps, not depth maps
   - see https://thousandyardstare.de/blog/generating-depth-images-in-blender-279.html for explanation
   -  use snippets/distance_to_depth.py with focal length from intrinsic matrix (intrinsic.txt)

## normals maps (normals_*.exr)
 - exr, 3 channels for xyz

## segmentation, 2D (segmap_*.npz)
 - numpy array, np.load(..)
 - 2 or 3 channels
   - 0: semantics
   - 1: instances

## extrinsic (campose_*.npy)
  - numpy "string" / json, np.load(..)
  - matrix, location, rotation_euler, fov_x, fov_y, (customprop_room_id, customprop_room_name)
  - blender coord. system!

## distance field (distance_field_*.df)
 - format: unsigned, untruncated distance field
 - voxel size: 5cm, [139, 104, 112]
 - use snippets/volume_reader.py -> read_df

## segmentation, 3d (segmentation_*.sem)
  - format: combined semantics & instances
    - eg: voxel value: 3001
    - semantics: value // 1000 -->  semantic label: 3
    - instances: value % 1000 --> instance label: 
  - use snippets/volume_reader.py -> read_semantics (to get both semantics and instances)



