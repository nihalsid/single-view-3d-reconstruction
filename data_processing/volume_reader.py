import numpy as np
import struct

typeNames = {
        'int8': 'b',
        'uint8': 'B',
        'int16': 'h',
        'uint16': 'H',
        'int32': 'i',
        'uint32': 'I',
        'int64': 'q',
        'uint64': 'Q',
        'float': 'f',
        'double': 'd',
        'char': 's'
        }


class BinaryReader(object):
    def __init__(self, filename):
        self.file = open(filename, 'rb')

    def read(self, typeName, times=1):
        typeFormat = typeNames[typeName.lower()]*times
        typeSize = struct.calcsize(typeFormat)
        value = self.file.read(typeSize)
        if typeSize != len(value):
            raise Exception
        return struct.unpack(typeFormat, value)

    def close(self):
        self.file.close()


def read_df(filename, down_scale_factor=1):
    reader = BinaryReader(filename)
    dimX, dimY, dimZ = reader.read('UINT64', 3)
    df = reader.read('float', dimX*dimY*dimZ)
    df = np.array(df, dtype=np.float32).reshape([dimX, dimY, dimZ], order='F')
    
    if down_scale_factor != 1:
        df = scale_down(df, down_scale_factor)

    return df

# ONLY SCALES TO A FACTOR OF TWO FOR NOW
# TODO: Implement scaling for factors other than 2
def scale_down(df, down_scale_factor):
    n_df = (df[:-1:2]+df[1::2])/2
    n_df = (n_df[:,:-1:2]+n_df[:,1::2])/2
    n_df = (n_df[:,:,:-1:2]+n_df[:,:,1::2])/2
    return n_df

def read_semantics(filename):
    reader = BinaryReader(filename)
    dimX, dimY, dimZ = reader.read('UINT64', 3)
    semantics = reader.read('uint16', dimX*dimY*dimZ)
    semantics = np.array(semantics, dtype=np.uint16).reshape([dimX, dimY, dimZ], order='F')

    perVoxelLabel = semantics / 1000
    perVoxelInstance = semantics % 1000

    return perVoxelLabel, perVoxelInstance


if __name__ == "__main__":
    from pathlib import Path
    from util.visualize import visualize_sdf
    input_df = Path("data") / "raw" / "overfit" / "00000" / "distance_field_0010.df"
    output_mesh = Path("data") / "visualizations" / "overfit" / "00000" / "mesh.obj"
    df = read_df(str(input_df))
    visualize_sdf(df, output_mesh, level=1.0)
