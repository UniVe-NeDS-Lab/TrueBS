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


def set_cover(viewsheds, n, k, type, tqdm_enable=True):
    covered_points = cuda.to_device(
        np.zeros(shape=(viewsheds.shape[0]), dtype=np.uint8))
    L = []
    for i in tqdm(range(n)) if tqdm_enable else range(n):
        r_max = 0
        r_min = np.inf
        jstar = -1
        ranks = cuda.device_array(shape=viewsheds.shape[1], dtype=np.float32)
        set_memory(ranks, 0)
        fi_rank_update[viewsheds.shape[1], 1](
            viewsheds, covered_points, ranks, k, fix_rank(type))
        if type != 'rlc':
            for j, ri in enumerate(ranks):
                if ri > r_max and j not in L:
                    r_max = ri
                    jstar = j
        else:
            for j, ri in enumerate(ranks):
                if ri < r_min and j not in L:
                    r_min = ri
                    jstar = j

        if jstar >= 0:
            L.append(jstar)
        update_coverage[viewsheds.shape[0], 1](
            viewsheds, covered_points, jstar, k)
    return L


def set_cover_faster(viewsheds, n, k, type, tqdm_enable=True):
    covered_points = cuda.to_device(
        np.zeros(shape=(viewsheds.shape[0]), dtype=np.uint8))
    L = []
    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(viewsheds.shape[0] / threadsperblock[1])
    blockspergrid_y = math.ceil(viewsheds.shape[1] / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    for i in tqdm(range(n)) if tqdm_enable else range(n):
        r = 0
        jstar = -1
        ranks = cuda.device_array(shape=viewsheds.shape[1], dtype=np.float32)
        set_memory(ranks, 0)
        if type == 'r1':
            fast_r1_rank_update[blockspergrid, threadsperblock](
                viewsheds, covered_points, ranks, k)
        elif type == 'r2':
            fast_r2_rank_update[blockspergrid, threadsperblock](
                viewsheds, covered_points, ranks, k)
        elif type == 'fi':
            sum1 = cuda.device_array(
                shape=viewsheds.shape[1], dtype=np.float32)
            sum2 = cuda.device_array(
                shape=viewsheds.shape[1], dtype=np.float32)
            set_memory(sum1, 0)
            set_memory(sum2, 0)
            fast_fi_rank_update[blockspergrid, threadsperblock](
                viewsheds, covered_points, sum1, sum2, k)
            fast_fi_calc[viewsheds.shape[0], 1](
                sum1, sum2, ranks, k, viewsheds.shape[1])
        for j, ri in enumerate(ranks):
            if ri > r and j not in L:
                r = ri
                jstar = j
        if jstar >= 0:
            L.append(jstar)
        update_coverage[viewsheds.shape[0], 1](
            viewsheds, covered_points, jstar, k)
    return L


@cuda.jit()
def fast_fi_calc(sum1, sum2, ranks, k, m):
    """
    Calculate the ranks of the viewsheds

    This function calculates the ranks of the viewsheds using the formula
    r(i) = k - sum(j) / sum(j) + sum(j) / sum(j)

    Parameters:
    sum1 (ndarray): sum of the viewsheds
    sum2 (ndarray): squared sum of the viewsheds
    ranks (ndarray): array to store the ranks
    k (): constant k
    """
    i = cuda.grid(1)
    if i < ranks.shape[0]:
        ranks[i] = (sum1[i]**2 / (m*sum2[i])) * (sum1[i]/(m*k))


@cuda.jit()
def fi_rank_update(viewsheds, covered_points, rank, k, type):
    mat = cuda.grid(1)
    if mat >= viewsheds.shape[1]:
        return
    summ = 0
    sumsq = 0
    sumrlc = 0
    m = viewsheds.shape[0]
    for i in range(m):
        viewshed_upd = viewsheds[i, mat] + covered_points[i]
        if viewshed_upd > k:  # Bounded sum to k
            viewshed_upd = k
        summ += viewshed_upd
        sumsq += viewshed_upd**2
        sumrlc += (k-viewshed_upd)**2
    if type == 0:
        rank[mat] = (summ**2 / (m*sumsq)) * (summ/(m*k))
    elif type == 1:
        rank[mat] = summ
    elif type == 2:
        rank[mat] = sumsq
    elif type == 3:
        rank[mat] = sumrlc


@cuda.jit()
def fast_r1_rank_update(viewsheds, covered_points, rank, k):
    j, i = cuda.grid(2)
    if j < viewsheds.shape[0] and i < viewsheds.shape[1]:
        viewshed_upd = viewsheds[j, i] + covered_points[j]
        if viewshed_upd > 0:  # Atomic op is costly, better do it only when there's something to sum
            if viewshed_upd > k:
                viewshed_upd = k
            cuda.atomic.add(rank, i, viewshed_upd)
        # else, viewshed_upd is 0 we don't do anything


@cuda.jit()
def fast_r2_rank_update(viewsheds, covered_points, rank, k):
    j, i = cuda.grid(2)
    if j < viewsheds.shape[0] and i < viewsheds.shape[1]:
        viewshed_upd = viewsheds[j, i] + covered_points[j]
        if viewshed_upd > 0:  # Atomic op is costly, better do it only when there's something to sum
            if viewshed_upd > k:
                viewshed_upd = k
            cuda.atomic.add(rank, i, viewshed_upd**2)
        # else, viewshed_upd is 0 we don't do anything


@cuda.jit()
def fast_fi_rank_update(viewsheds, covered_points, sum1, sum2, k):
    j, i = cuda.grid(2)
    if j < viewsheds.shape[0] and i < viewsheds.shape[1]:
        viewshed_upd = viewsheds[j, i] + covered_points[j]
        if viewshed_upd > 0:  # Atomic op is costly, better do it only when there's something to sum
            if viewshed_upd > k:
                viewshed_upd = k
            cuda.atomic.add(sum1, i, viewshed_upd)
            cuda.atomic.add(sum2, i, viewshed_upd**2)


@cuda.jit()
def update_coverage(viewsheds, covered_points, selected_mat, k):
    i = cuda.grid(1)
    covered_points[i] += viewsheds[i, selected_mat]
    if covered_points[i] > k:
        covered_points[i] = k  # Bounded sum to k


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
