from SetCover import *
import unittest
import sys
import numpy as np
sys.path.append('.')


class TestSetCover(unittest.TestCase):
    def test_update_coverage(self):
        viewsheds = np.zeros(shape=(10, 10), dtype=np.uint8)
        viewsheds[:, 0] = 1
        viewsheds[1, :] = 1
        covered_points = np.zeros(shape=(10), dtype=np.uint16)
        update_coverage[viewsheds.shape[0], 1](viewsheds, covered_points, 0, 2)
        res = np.ones(shape=10, dtype=np.uint16)
        assert np.array_equal(covered_points, res)
        update_coverage[viewsheds.shape[0], 1](viewsheds, covered_points, 1, 2)
        res[1] = 2
        assert np.array_equal(covered_points, res)
        update_coverage[viewsheds.shape[0], 1](viewsheds, covered_points, 1, 2)
        assert np.array_equal(covered_points, res)

    def test_fast_r1_rank_update(self):
        viewsheds = np.zeros(shape=(10, 5), dtype=np.uint8)
        viewsheds[:, 0] = 2
        viewsheds[1, :] = 1
        threadsperblock = (8, 8)
        blockspergrid_x = math.ceil(viewsheds.shape[0] / threadsperblock[1])
        blockspergrid_y = math.ceil(viewsheds.shape[1] / threadsperblock[0])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        covered_points = np.zeros(shape=(10), dtype=np.uint16)
        rank = np.zeros(shape=(5), dtype=np.float32)
        viewsheds_old = viewsheds.copy()
        fast_r1_rank_update[blockspergrid, threadsperblock](
            viewsheds, covered_points, rank, 2)
        assert np.array_equal(
            covered_points, np.zeros(shape=10, dtype=np.uint16))
        assert np.array_equal(rank, np.array([19., 1., 1., 1., 1.]))
        assert np.array_equal(viewsheds, viewsheds_old)

    def test_fast_r2_rank_update(self):
        viewsheds = np.zeros(shape=(10, 5), dtype=np.uint8)
        viewsheds[:, 0] = 2
        viewsheds[1, :] = 1
        threadsperblock = (8, 8)
        blockspergrid_x = math.ceil(viewsheds.shape[0] / threadsperblock[1])
        blockspergrid_y = math.ceil(viewsheds.shape[1] / threadsperblock[0])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        covered_points = np.zeros(shape=(10), dtype=np.uint16)
        rank = np.zeros(shape=(5), dtype=np.float32)
        viewsheds_old = viewsheds.copy()
        fast_r2_rank_update[blockspergrid, threadsperblock](
            viewsheds, covered_points, rank, 2)
        assert np.array_equal(
            covered_points, np.zeros(shape=10, dtype=np.uint16))
        assert np.array_equal(rank, np.array([37., 1., 1., 1., 1.]))
        assert np.array_equal(viewsheds, viewsheds_old)

    def test_fast_fi_rank_update(self):
        viewsheds = np.zeros(shape=(10, 5), dtype=np.uint8)
        viewsheds[:, 0] = 2
        viewsheds[1, :] = 1
        threadsperblock = (8, 8)
        blockspergrid_x = math.ceil(viewsheds.shape[0] / threadsperblock[1])
        blockspergrid_y = math.ceil(viewsheds.shape[1] / threadsperblock[0])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        covered_points = np.zeros(shape=(10), dtype=np.uint16)
        rank1 = np.zeros(shape=(5), dtype=np.float32)
        rank2 = np.zeros(shape=(5), dtype=np.float32)
        viewsheds_old = viewsheds.copy()
        fast_fi_rank_update[blockspergrid, threadsperblock](
            viewsheds, covered_points, rank1, rank2, 2)
        assert np.array_equal(
            covered_points, np.zeros(shape=10, dtype=np.uint16))
        assert np.array_equal(rank1, np.array([19., 1., 1., 1., 1.]))
        assert np.array_equal(rank2, np.array([37., 1., 1., 1., 1.]))
        assert np.array_equal(viewsheds, viewsheds_old)

    def test_fi_rank_update_rlc(self):
        viewsheds = np.zeros(shape=(10, 5), dtype=np.uint8)
        viewsheds[:, 0] = 2
        viewsheds[1, 0] = 1
        threadsperblock = (8, 8)
        blockspergrid_x = math.ceil(viewsheds.shape[0] / threadsperblock[1])
        blockspergrid_y = math.ceil(viewsheds.shape[1] / threadsperblock[0])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        covered_points = np.zeros(shape=(10), dtype=np.uint16)
        rank = np.zeros(shape=(5), dtype=np.float32)
        viewsheds_old = viewsheds.copy()
        fi_rank_update[blockspergrid, threadsperblock](
            viewsheds, covered_points, rank, 3, 3)
        assert np.array_equal(
            covered_points, np.zeros(shape=10, dtype=np.uint16))
        print(rank)
        assert np.array_equal(rank, np.array([13., 90., 90., 90., 90.]))
        assert np.array_equal(viewsheds, viewsheds_old)

    def test_fast_fi_calc(self):
        sum1 = np.array([1., 1., 1., 1., 1.])
        sum2 = np.array([1., 1., 1., 1., 1.])
        m = 10
        k = 2
        ranks = np.zeros_like(sum1)
        fast_fi_calc[len(ranks), 1](sum1, sum2, ranks, k, m)
        result = [(sum1[i]**2 / (m*sum2[i])) * (sum1[i]/(m*k))
                  for i in range(len(sum1))]
        assert np.array_equal(ranks, np.array(result))

    def test_setcover(self):
        viewsheds = np.zeros(shape=(10, 5), dtype=np.uint8)
        viewsheds[:, 0] = 1
        viewsheds[5:, 2] = 1
        viewsheds[7:, 3] = 1
        n = 3
        k = 3
        ranking_type = 'r1'
        L = set_cover(viewsheds, n, k, ranking_type, tqdm_enable=False)
        assert np.array_equal(L, [0, 2, 3])


if __name__ == '__main__':
    unittest.main()
