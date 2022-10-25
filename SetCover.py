from numba import cuda
import math
import numpy as np
from tqdm import tqdm
from kernels import memset_k_1d, memset_k
from numba.core.errors import NumbaPerformanceWarning
import warnings


warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def fix_rank(str):
    if str == 'fi':
        return 0
    elif str == 'r1':
        return 1
    elif str == 'r2':
        return 2
    elif str == 'rlc':
        return 3


def set_cover(viewsheds, n, tqdm_enable=True):
    covered_points = cuda.to_device(
        np.zeros(shape=(viewsheds.shape[0]), dtype=np.uint8))
    L = []
    for i in tqdm(range(n)) if tqdm_enable else range(n):
        r_max = 0
        jstar = -1
        ranks = cuda.device_array(shape=viewsheds.shape[1], dtype=np.float32)
        set_memory(ranks, 0)
        fi_rank_update[viewsheds.shape[1], 1](viewsheds, covered_points, ranks)
        for j, ri in enumerate(ranks):
            if ri >= r_max and j not in L:
                r_max = ri
                jstar = j
        if jstar >= 0:
            L.append(jstar)
        update_coverage[viewsheds.shape[0], 1](viewsheds, covered_points, jstar)
    return L

@cuda.jit()
def fi_rank_update(viewsheds, covered_points, rank):
    mat = cuda.grid(1)
    if mat >= viewsheds.shape[1]:
        return
    summ = 0
    m = viewsheds.shape[0]
    for i in range(m):
        viewshed_upd = viewsheds[i, mat] * (not bool(covered_points[i]))
        if viewshed_upd > 255:  # Bounded sum to 255 #TODO: use dtype.maxint
            viewshed_upd = 255
        summ += viewshed_upd
    
    rank[mat] = summ
    


@cuda.jit()
def update_coverage(viewsheds, covered_points, selected_mat):
    i = cuda.grid(1)
    covered_points[i] += viewsheds[i, selected_mat]
    if covered_points[i] > 255:
        covered_points[i] = 255  # Bounded sum to k


def set_memory(array, val):
    """
    Set a 2d or 1d ndarray to a given value using the custom kernel

    This function calls a kernel that set the memory space to a given value,
    should be fixed by using memset intead of the viewshed_kernel

    Parameters:
    array (ndarray): numba cuda array to be setted
    val (): value to set
    """
    if(len(array.shape) >= 2 and array.shape[1] >= 16):  # 2df
        threadsperblock = (32, 32)
        blockspergrid_y = int(math.ceil(array.shape[1] / threadsperblock[1]))
        blockspergrid_x = int(math.ceil(array.shape[0] / threadsperblock[0]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        memset_k[blockspergrid, threadsperblock](array, val)
    else:  # 1d or 2d smaller than 16
        blockspergrid_x = int(math.ceil(array.shape[0] / 32))
        memset_k_1d[blockspergrid_x, 32](array, val)
