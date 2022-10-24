import sys  # nopep8
sys.path.append('.')  # nopep8
from SetCover import *
from viewshed import Viewshed
import unittest
import numpy as np
import rasterio


class TestViewshed(unittest.TestCase):
    def test_single_viewshed(self):
        '''
        Calculate the viewshed using the single viewshed and
        compare it against the outcome of a classic Viewshed from Grass GIS.

        The output must be 95% similar.
        '''
        dataset_dsm = rasterio.open(
            'tests/test_data/firenze_scipione.tif')
        dsm = dataset_dsm.read(1)
        vs = Viewshed(1000)
        coords = dataset_dsm.index(1683417, 4848795)
        viewshed = vs.single_viewshed(dsm, coords, 2, 2, 0).copy_to_host()
        expected_vs = rasterio.open(
            'tests/test_data/firenze_scipione_1683417_4848795_2m_2m.grass.tif')
        gpu_vs = expected_vs.read(1)
        error = np.absolute(gpu_vs.astype(np.int16) -
                            viewshed.astype(np.int16)).sum()
        size = dsm.shape[0]*dsm.shape[1]
        rel_error = error/size
        assert rel_error < 0.05

    def test_parallel_translated_viewsheds_single(self):
        vs = Viewshed(1000)
        dataset_dsm = rasterio.open(
            'tests/test_data/firenze_scipione.tif')
        dsm = dataset_dsm.read(1)
        coords = dataset_dsm.index(1683417, 4848795)
        poi_coords = [coords]
        poi_elev = 2
        tgt_elev = 2
        road_mask = np.ones_like(dsm, dtype=np.uint8)
        transmat, inv_transmat = vs.gen_translation_matrix(road_mask)
        t_vs = vs.parallel_viewsheds_translated(
            dsm, poi_coords, transmat, np.count_nonzero(road_mask), poi_elev, tgt_elev, 0).copy_to_host()
        viewshed = vs.translate_viewshed(t_vs[:, 0], inv_transmat, dsm.shape)

        res = vs.single_viewshed(
            dsm, poi_coords[0], poi_elev, tgt_elev, 0).copy_to_host()

        assert np.array_equal(viewshed, res*road_mask)

    def test_parallel_translated_viewsheds_multiple(self):
        vs = Viewshed(1000)
        dataset_dsm = rasterio.open(
            'tests/test_data/firenze_scipione.tif')
        dsm = dataset_dsm.read(1)
        coords_1 = dataset_dsm.index(1683417, 4848795)
        coords_2 = dataset_dsm.index(1683317, 4848815)
        poi_coords = [coords_1, coords_2]*50
        poi_elev = 2
        tgt_elev = 2
        road_mask = np.ones_like(dsm, dtype=np.uint8)
        transmat, inv_transmat = vs.gen_translation_matrix(road_mask)
        t_vs = vs.parallel_viewsheds_translated(
            dsm, poi_coords, transmat, np.count_nonzero(road_mask), poi_elev, tgt_elev, 0).copy_to_host()
        result = np.zeros(
            shape=(dsm.shape[0], dsm.shape[1], len(poi_coords)), dtype=np.uint8)
        expected_result = np.zeros_like(result)
        for i in range(len(poi_coords)):
            result[:, :, i] = vs.translate_viewshed(
                t_vs[:, i], inv_transmat, dsm.shape)*road_mask
            expected_result[:, :, i] = vs.single_viewshed(
                dsm, poi_coords[i], poi_elev, tgt_elev, 0).copy_to_host()

        assert np.array_equal(result, expected_result)

    def test_parallel_cumulative_buildings_vs(self):
        vs = Viewshed(1000)
        dataset_dsm = rasterio.open(
            'tests/test_data/firenze_scipione.tif')
        dsm = dataset_dsm.read(1)
        # dsm = np.zeros((10,10), dtype=np.float32)
        poi_elev = 2
        tgt_elev = 2
        coords_1 = dataset_dsm.index(1683417, 4848795)
        coords_2 = dataset_dsm.index(1683317, 4848815)
        coords_lists = [[coords_1, coords_2], [coords_1], [coords_2]]
        road_mask = np.ones_like(dsm, dtype=np.uint8)
        transmat, inv_transmat = vs.gen_translation_matrix(road_mask)

        cum_viewshed = vs.parallel_cumulative_buildings_vs(
            dsm, transmat, np.count_nonzero(road_mask), coords_lists, poi_elev, tgt_elev, 0).copy_to_host()

        viewsheds_res = np.zeros_like(cum_viewshed)
        for i, coords in enumerate(coords_lists):
            viewshed = vs.parallel_viewsheds_translated(
                dsm, coords, transmat, np.count_nonzero(road_mask), poi_elev, tgt_elev, 0).copy_to_host()
            for j in range(len(coords)):
                viewsheds_res[:, i] = np.bitwise_or(
                    viewsheds_res[:, i], viewshed[:, j])

        assert np.array_equal(viewsheds_res, cum_viewshed)


if __name__ == '__main__':
    unittest.main()
