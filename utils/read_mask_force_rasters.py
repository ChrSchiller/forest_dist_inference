import numpy as np
import rasterio

def read_mask_force_rasters(file_path, mask):
    with rasterio.open(file_path) as src:
        rast = src.read()
    ### convert numpy array to memory-mapped file
    memmap = np.memmap('rast.mmap', rast.dtype, mode='w+', shape = rast.shape)
    # write the numpy array to the memory-mapped file
    memmap[:, :, :] = rast
    ### apply forest mask
    memmap = memmap * mask
    ### remove all -9999 values coming from quality screening (cloud mask, etc.)
    memmap[memmap == -9999] = np.nan
    ### remove rast from memory
    rast = None

    return memmap.transpose([1, 2, 0])