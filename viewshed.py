import numpy as np
from numba import cuda
import math
from kernels import memset_k_1d, viewshed_k, threads_n, par_blocks, memset_k,  parallel_viewshed_t_k, logical_or_axis_k, CORNER_OVERLAP
from tqdm import tqdm


class Viewshed():
    def __init__(self,  max_dist=0):
        self.max_dist = max_dist
        self.ctx = cuda.current_context()
        # self.reset_memory()

    def get_memory(self):
        return self.ctx.get_memory_info().total

    def reset_memory(self):
        # self.ctx.memory_manager.reset()
        pass

    def single_viewshed(self, raster_np, poi_coord, poi_elev, tgt_elev, poi_elev_type=0):
        """
        Run the  single viewshed kernel

        Calculate the viewshed using Osterman algorithm from a point to the whole DSM.

        Parameters:
        poi_elev (int): height of the observer above the DSM
        tgt_elev (int): height of the target above the DSM
        """
        self.reset_memory()
        raster = raster_np
        dsm_global_mem = cuda.to_device(raster_np)  # transfer dsm
        out_global_mem = cuda.device_array(
            shape=raster_np.shape, dtype=np.uint8)
        self.set_memory(out_global_mem, 0)
        # calculate block size and thread number
        blocks_landscape = (raster.shape[0] +
                            2*raster.shape[0]/threads_n +
                            CORNER_OVERLAP +
                            threads_n-1)/threads_n
        block_upright = (raster.shape[1] +
                         2*raster.shape[1]/threads_n +
                         CORNER_OVERLAP +
                         threads_n-1)/threads_n
        blocks_n = max(block_upright, blocks_landscape)
        blockspergrid = (int(blocks_n), 4)
        threadsperblock = (threads_n, 1)
        viewshed_k[blockspergrid, threadsperblock](dsm_global_mem,
                                                   out_global_mem,
                                                   np.int32(poi_coord),
                                                   np.int16(self.max_dist),
                                                   np.int16(1),
                                                   np.int16(1),
                                                   np.float32(poi_elev),
                                                   np.float32(tgt_elev),
                                                   poi_elev_type)
        return out_global_mem

    def parallel_viewsheds_translated(self, raster_np, poi_coords, translation_matrix, n_points, poi_elev, tgt_elev, poi_elev_type):
        '''
        This function calculates n_vs viewsheds parallely. Each viewshed is stored as a line and the corresponding translation is given by translation_matrix

        Parameters:
        raster_np : numpy array containing the Digital Elevation Model
        poi_coords: list of coordinates for the observer point (either [x,y] or [x,y,z] depending on poi_elev_type)
        translation_matrix: matrix with shape equal to raster_np, each cell can contains either 0 (ignore the value) or an index between 1:npoints+1
        n_points: the total number of target points
        poi_elev: the height of the observer point above the DEM (if poi_elev_type == 0 this should be 0)
        tgt_elev: the height of the target point above the DEM
        poi_elev_type: this value specifies if the height of the point of interest must be taken from poi_coords[:,2] or from poi_elev
        '''
        assert n_points == translation_matrix.max()
        if poi_elev_type == 1:
            assert len(raster_np.shape) == 3
            assert poi_elev == 0
        self.reset_memory()
        n_vs = len(poi_coords)
        # allocate and initialize memory to store viewsheds (to 0s)
        out_global_mem = cuda.device_array(
            shape=(n_points, n_vs), dtype=np.uint8)
        self.set_memory(out_global_mem, 0)

        # calculate block size and thread number
        blocks_landscape = (raster_np.shape[0] +
                            2*raster_np.shape[0]/threads_n +
                            CORNER_OVERLAP +
                            threads_n-1)/threads_n
        block_upright = (raster_np.shape[1] +
                         2*raster_np.shape[1]/threads_n +
                         CORNER_OVERLAP +
                         threads_n-1)/threads_n

        blocks_n = max(block_upright, blocks_landscape)
        # by keeping the number of viewsheds on the x we can have up to 2^32 -1
        blockspergrid = (n_vs, int(blocks_n), 4)
        threadsperblock = (1, threads_n, 1)
        parallel_viewshed_t_k[blockspergrid,
                              threadsperblock](raster_np,
                                               out_global_mem,
                                               translation_matrix,
                                               cuda.to_device(poi_coords),
                                               np.int16(self.max_dist),
                                               np.int16(1),
                                               np.int16(1),
                                               np.float32(poi_elev),
                                               np.float32(tgt_elev),
                                               poi_elev_type)
        # not copying it
        return out_global_mem

    def set_memory(self, array, val):
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
            blockspergrid_y = int(
                math.ceil(array.shape[1] / threadsperblock[1]))
            blockspergrid_x = int(
                math.ceil(array.shape[0] / threadsperblock[0]))
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            memset_k[blockspergrid, threadsperblock](array, val)
        else:  # 1d or 2d smaller than 16
            blockspergrid_x = int(math.ceil(array.shape[0] / 32))
            memset_k_1d[blockspergrid_x, 32](array, val)

    def parallel_cumulative_buildings_vs(self, raster, translation_matrix, n_points, coords_lists, poi_elev, tgt_elev, poi_elev_type):
        self.ctx.memory_manager.reset()
        output = np.zeros(shape=(n_points, len(coords_lists)), dtype=np.uint8)
        output_cuda = cuda.to_device(output)
        for idx, c_list in enumerate(tqdm(coords_lists)):
            if c_list:
                dsm_global_mem = cuda.to_device(raster)  # transfer dsm
                cu_translation_matrix = cuda.to_device(
                    translation_matrix)  # transfer dsm
                building_n = len(c_list)
                cu_coords = cuda.to_device(c_list)
                out_global_mem = cuda.device_array(
                    shape=(n_points, building_n), dtype=np.uint8)
                self.set_memory(out_global_mem, 0)
                # calculate block size and thread number
                blocks_landscape = (raster.shape[0] +
                                    2*raster.shape[0]/threads_n +
                                    CORNER_OVERLAP +
                                    threads_n-1)/threads_n
                block_upright = (raster.shape[1] +
                                 2*raster.shape[1]/threads_n +
                                 CORNER_OVERLAP +
                                 threads_n-1)/threads_n

                blocks_n = max(block_upright, blocks_landscape)
                blockspergrid = (building_n, int(blocks_n), 4)
                threadsperblock = (1, threads_n, 1)
                parallel_viewshed_t_k[blockspergrid, threadsperblock](dsm_global_mem,
                                                                      out_global_mem,
                                                                      cu_translation_matrix,
                                                                      cu_coords,
                                                                      np.int16(
                                                                          self.max_dist),
                                                                      np.int16(
                                                                          1),
                                                                      # TODO: fix variable resolution
                                                                      np.int16(
                                                                          1),
                                                                      np.float32(
                                                                          poi_elev),
                                                                      np.float32(
                                                                          tgt_elev),
                                                                      poi_elev_type)

                logical_or_axis_k[n_points, 1](
                    out_global_mem, output_cuda, idx)
                self.set_memory(out_global_mem, 0)

        return output_cuda

    def gen_translation_matrix(self, road_mask):
        area = int(road_mask.sum() + 1)
        translation_matrix = np.zeros(shape=road_mask.shape, dtype=np.uint32)
        inverse_matrix = np.zeros(shape=(area, 2), dtype=np.uint32)
        count = 1
        for i in range(road_mask.shape[0]):
            for j in range(road_mask.shape[1]):
                if road_mask[i, j]:
                    translation_matrix[i, j] = count
                    inverse_matrix[count] = [i, j]
                    count += 1

        return translation_matrix, inverse_matrix

    def translate_viewshed(self, linear_vs, inverse_matrix, size):
        viewshed = np.zeros(shape=size, dtype=np.uint8)
        for idx, e in enumerate(inverse_matrix[1:]):
            viewshed[e[0], e[1]] = linear_vs[idx]
        return viewshed
