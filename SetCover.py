from numba import cuda
import math
import numpy as np
from tqdm import tqdm
from kernels import *
from numba.core.errors import NumbaPerformanceWarning
import warnings
import cupy
import time

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


def set_cover(viewsheds, n, k, type, vg_np, tqdm_enable=True):
    covered_points = cuda.to_device(
        np.zeros(shape=(viewsheds.shape[0]), dtype=np.uint8))
    L = []
    n = vg_np.shape[0]
    c_graph = cuda.to_device(vg_np)
    c_subgraph = cuda.to_device(np.zeros(shape=(n,n,n), dtype=np.uint8))
    c_temp = cuda.to_device(np.zeros(shape=(n,n,n), dtype=np.uint8))
    c_lapl = cuda.to_device(np.zeros(shape=(n,n,n), dtype=np.int32))
    for i in tqdm(range(n)) if tqdm_enable else range(n):
        r_max = 0
        r_min = np.inf
        jstar = -1
        ranks = cuda.device_array(shape=viewsheds.shape[1], dtype=np.float32)
        set_memory(ranks, 0)
        fi_rank_update[viewsheds.shape[1], 1](
            viewsheds, covered_points, ranks, k, fix_rank(type))
        cu_add_all_node(c_graph, c_temp, c_subgraph)

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
        if i > 10:
            eig = cu_connectivity(c_subgraph, c_lapl)
            break
            #print(eig)
        
        if jstar >= 0:
            L.append(jstar)
            copy_mat[c_graph.shape, (1,1)](jstar, c_temp, c_subgraph)
        #update graph
        
        #update ground coverage
        update_coverage[viewsheds.shape[0], 1](
            viewsheds, covered_points, jstar, k)

    return L


@cuda.jit()
def copy_row(source, dest):
    i,j = cuda.grid(2)
    if j<source.shape[0] and i<source.shape[0]:
        dest[i,i,j] = source[i,j]

@cuda.jit()
def bitwise_rowcol(source, dest):
    i,j = cuda.grid(2)
    if j<source.shape[0] and i<source.shape[0]:
        dest[i,i,j] = dest[i, j, i] = source[i, i, j] & source[i, j, i] 

def cu_add_all_node(source, temp, subgraph):
    copy_row[source.shape, (1,1)](source, temp)
    bitwise_rowcol[source.shape, (1,1)](temp, subgraph)


def cu_connectivity(subgraph, laplacian):
    t=time.time()
    cu_laplacian[subgraph.shape, (1,1)](subgraph, laplacian)
    t2 = time.time()
    #l = laplacian.copy_to_host()
    t3 = time.time()
    la = cupy.array(laplacian, laplacian.dtype, copy=False)
    eigv = cupy.linalg.eigvalsh(la)
    t4 = time.time()
    print(f'lapl: {t2-t}, copy1: {t3-t2},  copy2: {t4-t3}, eigv: {time.time()-t4}')
    #print(eigv)
    #cc = np.zeros(shape=subgraph.shape[0])
    # for i in range(subgraph.shape[0]):
    #     for j in range(subgraph.shape[1]):
    #         if eigv[i,j] > 1e-10:
    #             cc[i] = j
    #             break
    #return cc
        
    

@cuda.jit()
def cu_laplacian(graph, lapl):
    i,j = cuda.grid(2) #i is the graph index, j is the row index
    if j<graph.shape[0] and i<graph.shape[0]:
        d = 0
        for k in range(graph.shape[0]): #iterate on the row
            d+=graph[i, j, k]
            lapl[i,j,k] = -1*graph[i,j,k] #set the negative values
        lapl[i, j, j] = d  #Set the diagonal equal to the degree



@cuda.jit()
def copy_mat(m_i, mat1, mat2):
    i,j = cuda.grid(2)
    if j<mat1.shape[1] and i<mat1.shape[2]:
        for k in range(mat1.shape[0]):
            if k != m_i:
                mat1[k,i,j] = mat1[m_i, i, j]
                mat2[k,i,j] = mat2[m_i, i, j]


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
