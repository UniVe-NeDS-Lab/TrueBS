from viewshed import Viewshed
from SetCover import *
import unittest
import sys
import numpy as np
import warnings
sys.path.append('.')


class TestViewshed(unittest.TestCase):
    def test_single_viewshed(self):
        vs = Viewshed(1000)
        dsm = np.zeros((10, 10))
        poi_coords = [0, 0]
        poi_elev, tgt_elev = (1, 1)
        viewshed = vs.single_viewshed(
            dsm, poi_coords, poi_elev, tgt_elev, 0).copy_to_host()
        assert np.array_equal(viewshed, np.ones_like(dsm))

        dsm[:, 5] = 2
        viewshed = vs.single_viewshed(
            dsm, poi_coords, poi_elev, tgt_elev, 0).copy_to_host()
        result = np.ones_like(dsm)
        result[:, 6:] = 0
        assert np.array_equal(viewshed, result)
        dsm[:, :] = 0
        viewshed = vs.single_viewshed(dsm, poi_coords, 1, -2, 0).copy_to_host()

        result = np.zeros_like(dsm)
        result[poi_coords[0], poi_coords[1]] = 1
        assert np.array_equal(viewshed, result)

    def test_road_mask(self):
        vs = Viewshed(1000)
        dsm = np.zeros((10, 10))
        dsm[:, 5] = 2
        road_mask = np.zeros_like(dsm)
        road_mask[1, :] = 1
        road_mask[3, :] = 1
        transmat, inv_transmat = vs.gen_translation_matrix(road_mask)
        res1 = np.where(transmat > 0, 1, 0)
        assert np.array_equal(res1, road_mask)
        for coords in inv_transmat[1:]:
            assert road_mask[coords[0], coords[1]] == 1.0

    def test_parallel_translated_viewsheds_single(self):
        vs = Viewshed(1000)
        dsm = np.zeros((10, 10), dtype=np.float32)
        poi_coords = [[1, 0]]  # , [0,5], [5,0], [5,5]]
        poi_elev = 1
        tgt_elev = 1
        road_mask = np.zeros_like(dsm, dtype=np.uint8)
        road_mask[1, :] = 1
        road_mask[3, :] = 1
        transmat, inv_transmat = vs.gen_translation_matrix(road_mask)
        dsm[:, 5] = 2
        t_vs = vs.parallel_viewsheds_translated(
            dsm, poi_coords, transmat, road_mask.sum(), poi_elev, tgt_elev, 0).copy_to_host()
        viewshed = vs.translate_viewshed(t_vs[:, 0], inv_transmat, dsm.shape)

        res = vs.single_viewshed(
            dsm, poi_coords[0], poi_elev, tgt_elev, 0).copy_to_host()

        assert np.array_equal(viewshed, res*road_mask)

    def test_parallel_translated_viewsheds_multiple(self):
        vs = Viewshed(1000)
        dsm = np.zeros((10, 10), dtype=np.float32)
        poi_coords = [[1, 0], [0, 5], [0, 5], [5, 5]]
        poi_elev = 1
        tgt_elev = 1
        road_mask = np.zeros_like(dsm, dtype=np.uint8)
        road_mask[1, :] = 1
        road_mask[3, :] = 1
        transmat, inv_transmat = vs.gen_translation_matrix(road_mask)
        dsm[:, 5] = 2

        viewsheds_res = np.zeros(
            shape=(dsm.shape[0], dsm.shape[1], len(poi_coords)))
        for idx, c in enumerate(poi_coords):
            viewsheds_res[:, :, idx] = vs.single_viewshed(
                dsm, c, poi_elev, tgt_elev, 0).copy_to_host()
            viewsheds_res[:, :, idx] *= road_mask

        t_vs = vs.parallel_viewsheds_translated(
            dsm, poi_coords, transmat, road_mask.sum(), poi_elev, tgt_elev, 0).copy_to_host()

        viewsheds = np.zeros(
            shape=(dsm.shape[0], dsm.shape[1], len(poi_coords)))
        for idx, c in enumerate(poi_coords):
            viewsheds[:, :, idx] = vs.translate_viewshed(
                t_vs[:, idx], inv_transmat, dsm.shape)

        assert np.array_equal(viewsheds, viewsheds_res)

    def test_parallel_translated_viewsheds_single55(self):
        vs = Viewshed(1000)
        dsm = np.zeros((10, 10), dtype=np.float32)
        poi_coords = [[5, 5]]
        poi_elev = 1
        tgt_elev = 1
        road_mask = np.zeros_like(dsm, dtype=np.uint8)
        road_mask[1, :] = 1
        road_mask[3, :] = 1
        transmat, inv_transmat = vs.gen_translation_matrix(road_mask)
        dsm[:, 5] = 2
        t_vs = vs.parallel_viewsheds_translated(
            dsm, poi_coords, transmat, road_mask.sum(), poi_elev, tgt_elev, 0).copy_to_host()
        viewshed = vs.translate_viewshed(t_vs[:, 0], inv_transmat, dsm.shape)

        res = vs.single_viewshed(
            dsm, poi_coords[0], poi_elev, tgt_elev, 0).copy_to_host()

        assert np.array_equal(viewshed, res*road_mask)

    def test_parallel_cumulative_buildings_vs(self):
        vs = Viewshed(1000)
        dsm = np.random.choice([0, 2], size=(100, 100), p=[2./4, 2./4])
        # dsm = np.zeros((10,10), dtype=np.float32)
        poi_coords = [[5, 5]]
        dsm[:, 5] = 2
        dsm[5, :] = 2
        poi_elev = 1
        tgt_elev = 1
        coords_lists = [[[0, 0], [6, 6]], [[6, 6]]]
        road_mask = np.zeros_like(dsm, dtype=np.uint8)
        road_mask[1, :] = 1
        road_mask[8, :] = 1
        transmat, inv_transmat = vs.gen_translation_matrix(road_mask)

        cum_viewshed = vs.parallel_cumulative_buildings_vs(
            dsm, transmat, road_mask.sum(), coords_lists, poi_elev, tgt_elev, 0).copy_to_host()

        viewsheds_res = np.zeros_like(cum_viewshed)
        for i, coords in enumerate(coords_lists):
            viewshed = vs.parallel_viewsheds_translated(
                dsm, coords, transmat, road_mask.sum(), poi_elev, tgt_elev, 0).copy_to_host()
            for j in range(len(coords)):
                viewsheds_res[:, i] = np.bitwise_or(
                    viewsheds_res[:, i], viewshed[:, j])

        assert np.array_equal(viewsheds_res, cum_viewshed)

    def test_memset_2d(self):
        vs = Viewshed(1000)
        array = cuda.device_array(shape=(1000, 1000), dtype=np.uint32)
        vs.set_memory(array, 0)
        res = array.copy_to_host()
        assert np.array_equal(res, np.zeros_like(res))

    def test_memset_1d(self):
        vs = Viewshed(1000)
        array = cuda.device_array(shape=(1000), dtype=np.uint32)
        vs.set_memory(array, 0)
        res = array.copy_to_host()
        assert np.array_equal(res, np.zeros_like(res))

    def test_memset_small(self):
        vs = Viewshed(1000)
        array = cuda.device_array(shape=(10), dtype=np.uint32)
        vs.set_memory(array, 0)
        res = array.copy_to_host()
        assert np.array_equal(res, np.zeros_like(res))


if __name__ == '__main__':
    unittest.main()
