import numpy as np
from numba import cuda, types
import sys
import math
from time import time
from kernels import viewshed_k, threads_n, sum_k, memset_k, take_values, bool_k, parallel_viewshed_t_k, logical_or_axis_k, CORNER_OVERLAP
from tqdm import tqdm
from pprint import pprint



class Viewshed():
    def __init__(self,  max_dist=0):
        self.max_dist = max_dist
        self.ctx = cuda.current_context()
    
    def get_memory(self):
        return cuda.current_context().get_memory_info().total
    
    def prepare_cumulative_viewshed_bs(self, raster_np):
        """
        Prepare the cuda device for the upcoming cumulative viewshed computation.

        Allocate and copy the raster memory space, allocate the global memory
        for the computation of the single viewshed and the global memory for the
        cumulative adding

        Parameters:
        raster_np ([:][:]): 2d matrix containing the DSM of the area

        """
        self.raster = raster_np
        self.dsm_global_mem = cuda.to_device(raster_np) # transfer dsm
        self.out_global_mem = cuda.device_array(shape=raster_np.shape, dtype=np.uint8)
        self.cumulative_global_mem = cuda.device_array(shape=raster_np.shape, dtype=np.uint16) #with 16bit we can have up to 65k buildings
        self.cumulative_bool_mem = cuda.device_array(shape=raster_np.shape, dtype=np.uint16) #with 16bit we can have up to 65k buildings
        self.set_memory(self.cumulative_global_mem, 0)
        self.set_memory(self.out_global_mem, 0)
        self.set_memory(self.cumulative_bool_mem, 0)
    
    def single_viewshed(self, raster_np, poi_coord, poi_elev, tgt_elev, poi_elev_type=0):
        self.ctx.memory_manager.reset()
        self.raster = raster_np
        self.dsm_global_mem = cuda.to_device(raster_np) # transfer dsm
        self.out_global_mem = cuda.device_array(shape=raster_np.shape, dtype=np.uint8)
        self.set_memory(self.out_global_mem, 0)
        self._run_viewshed(poi_coord, poi_elev, tgt_elev, poi_elev_type)
        return self.out_global_mem.copy_to_host()

    def _run_viewshed(self, poi_coord, poi_elev, tgt_elev, poi_elev_type=0):
        """
        Run the  single viewshed kernel

        Calculate the viewshed using Osterman algorithm from a point to the whole DSM.

        Parameters:
        poi_elev (int): height of the observer above the DSM
        tgt_elev (int): height of the target above the DSM
        """
        #calculate block size and thread number
        blocks_landscape = (self.raster.shape[0] +
                            2*self.raster.shape[0]/threads_n +
                            CORNER_OVERLAP +
                            threads_n-1)/threads_n
        block_upright = (self.raster.shape[1] +
                         2*self.raster.shape[1]/threads_n +
                         CORNER_OVERLAP +
                         threads_n-1)/threads_n
        blocks_n = max(block_upright, blocks_landscape)
        blockspergrid = (int(blocks_n), 4)
        threadsperblock = (threads_n, 1)
        viewshed_k[blockspergrid, threadsperblock](self.dsm_global_mem,
                                                   self.out_global_mem,
                                                   np.int32(poi_coord),
                                                   np.int16(self.max_dist),
                                                   np.int16(1),
                                                   np.int16(1),
                                                   np.float32(poi_elev),
                                                   np.float32(tgt_elev),
                                                   poi_elev_type)

    def parallel_viewsheds_translated(self, raster_np, poi_coords, translation_matrix, n_points, poi_elev, tgt_elev, poi_elev_type):
        self.ctx.memory_manager.reset()
        self.raster = raster_np
        self.dsm_global_mem = cuda.to_device(raster_np) # transfer dsm
        self.translation_matrix = cuda.to_device(translation_matrix) # transfer dsm
        self.building_n =  len(poi_coords)
        self.coords = cuda.to_device(poi_coords)
        self.out_global_mem = cuda.device_array(shape=(n_points, self.building_n), dtype=np.uint8)
        self.set_memory(self.out_global_mem, 0)
        #calculate block size and thread number
        blocks_landscape = (self.raster.shape[0] +
                            2*self.raster.shape[0]/threads_n +
                            CORNER_OVERLAP +
                            threads_n-1)/threads_n
        block_upright = (self.raster.shape[1] +
                         2*self.raster.shape[1]/threads_n +
                         CORNER_OVERLAP +
                         threads_n-1)/threads_n
        
        blocks_z = self.building_n/threads_n
    
        blocks_n = max(block_upright, blocks_landscape)
        blockspergrid = (int(blocks_n), 4, math.ceil(self.building_n/2))
        threadsperblock = (threads_n, 1, 2)
        parallel_viewshed_t_k[blockspergrid, threadsperblock](self.dsm_global_mem,
                                                            self.out_global_mem,
                                                            self.translation_matrix,
                                                            self.coords,
                                                            np.int16(self.max_dist),
                                                            np.int16(1),
                                                            np.int16(1), #TODO: fix variable resolution
                                                            np.float32(poi_elev),
                                                            np.float32(tgt_elev),
                                                            poi_elev_type)

    def set_memory(self, array, val):
        """
        Set a 2d or 1d ndarray to a given value using the custom kernel

        This function calls a kernel that set the memory space to a given value,
        should be fixed by using memset intead of the viewshed_kernel

        Parameters:
        array (ndarray): numba cuda array to be setted
        val (): value to set
        """

        if(len(array.shape)>=2 and array.shape[1] >= 16): #2df
            threadsperblock = (16, 16)
            blockspergrid_y = int(math.ceil(array.shape[1] / threadsperblock[1]))
        else:                     #1d or 2d smaller than 16
            threadsperblock = (16, 1)
            blockspergrid_y = 1
        blockspergrid_x = int(math.ceil(array.shape[0] / threadsperblock[0]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        memset_k[blockspergrid, threadsperblock](array, val)

    def sum_results(self):
        """
        Sum the results of the viewshed computation on another memory space and
        set the original one to 0

        This function calls a kernel that set the memory space to a given value

        Parameters:
        array (ndarray): numba cuda array to be setted
        val (): value to set
        """

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(self.out_global_mem.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(self.out_global_mem.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        sum_k[blockspergrid, threadsperblock](self.out_global_mem, self.cumulative_global_mem)
    
    def booleanize_results(self):
        """
        Booleanize the result of the cumulative_vs into 0s and 1s into a new memory and reset the original one

        This function calls a kernel that set the memory space to a given value

        Parameters:
        array (ndarray): numba cuda array to be setted
        val (): value to set
        """

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(self.out_global_mem.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(self.out_global_mem.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        bool_k[blockspergrid, threadsperblock](self.cumulative_global_mem, self.cumulative_bool_mem)

    def parallel_cumulative_buildings_vs(self, raster, translation_matrix, n_points, coords_lists, poi_elev, tgt_elev, poi_elev_type):
        self.ctx.memory_manager.reset()
        output = np.zeros(shape=(n_points, len(coords_lists)), dtype=np.uint8)
        output_cuda = cuda.to_device(output)
        for idx, c_list in enumerate(tqdm(coords_lists)):
            if c_list:
                dsm_global_mem = cuda.to_device(raster) # transfer dsm
                cu_translation_matrix = cuda.to_device(translation_matrix) # transfer dsm
                building_n =  len(c_list)
                cu_coords = cuda.to_device(c_list)
                out_global_mem = cuda.device_array(shape=(n_points, building_n), dtype=np.uint8)
                self.set_memory(out_global_mem, 0)
                #calculate block size and thread number
                blocks_landscape = (raster.shape[0] +
                                    2*raster.shape[0]/threads_n +
                                    CORNER_OVERLAP +
                                    threads_n-1)/threads_n
                block_upright = (raster.shape[1] +
                                2*raster.shape[1]/threads_n +
                                CORNER_OVERLAP +
                                threads_n-1)/threads_n
            
                blocks_n = max(block_upright, blocks_landscape)
                blockspergrid = (int(blocks_n), 4,  math.ceil(building_n/2))
                threadsperblock = (threads_n, 1, 2)
                parallel_viewshed_t_k[blockspergrid, threadsperblock](dsm_global_mem,
                                                                    out_global_mem,
                                                                    cu_translation_matrix,
                                                                    cu_coords,
                                                                    np.int16(self.max_dist),
                                                                    np.int16(1),
                                                                    np.int16(1), #TODO: fix variable resolution
                                                                    np.float32(poi_elev),
                                                                    np.float32(tgt_elev),
                                                                    poi_elev_type)
               
                logical_or_axis_k[n_points, 1](out_global_mem, output_cuda, idx)
                self.set_memory(out_global_mem, 0)

        self.out_global_mem = output_cuda
    
    def gen_translation_matrix(self, road_mask):
        area = int(road_mask.sum() + 1)
        translation_matrix = np.zeros(shape=road_mask.shape, dtype=np.uint32)
        inverse_matrix = np.zeros(shape=(area, 2), dtype=np.uint32)
        count = 1
        for i in range(road_mask.shape[0]):
            for j in range(road_mask.shape[1]):
                if road_mask[i,j]:
                    translation_matrix[i,j] = count
                    inverse_matrix[count] = [i, j]
                    count += 1 
        
        return translation_matrix, inverse_matrix
    
    def translate_viewshed(self, linear_vs, inverse_matrix, size):
        viewshed=np.zeros(shape=size, dtype=np.uint8)
        for idx, e in enumerate(inverse_matrix[1:]):
            viewshed[e[0], e[1]] = linear_vs[idx]
        return viewshed