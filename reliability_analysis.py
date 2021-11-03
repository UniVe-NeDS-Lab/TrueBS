import csv
import configargparse
import numpy as np
from tqdm import tqdm
from pprint import pprint
import os
import glob
import math as m
import scipy.stats
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import random

max_antper_bs = 60
antper_bs_cost = 5


def conf_int(data, axis=0, alpha=0.95):
    n = 1
    for ax in axis:
        n *= data.shape[ax]
    sem = np.std(data, axis=axis, ddof=1) / np.sqrt(n)
    return sem * scipy.stats.t.ppf((1 + alpha) / 2., n-1)


class Reliability():
    def __init__(self, base_dir, comuni, sub_area_id, ratio, denss, c_ant, c_build, k, rt, ncovs, strategy, cached):
        self.srid = 3003  # TODO: check if it is metric
        self.crs = "EPSG:%4d" % (self.srid)
        self.base_dir = base_dir
        self.dens = denss
        self.comuni = comuni
        self.ratio = ratio
        self.sub_area_id = sub_area_id
        self.c_ant = c_ant
        self.c_build = c_build
        self.cached = cached
        self.k = k
        self.rt = rt
        self.ncovs = ncovs
        self.strategy = strategy
        self.viewsheds = []
        self.default_cycler = (cycler(color=['r', 'g', 'b', 'y', 'black', 'orange']) +
                               cycler(linestyle=['-', '--', '-', '--', '-', '--']))

    # def load_data(self):

    #     for vdx, (area, sa_id) in enumerate(itertools.product(self.comuni, self.sub_area_id)):
    #         vsss = []
    #         for rt in self.rt:
    #             vss = []
    #             for k in self.k:
    #                 try:
    #                     folder = f'{self.base_dir}/{area}/{self.strategy}/{sa_id}/{rt}/{k}/{self.ratio}/{self.dens}/'
    #                     vs = np.load(f'{folder}/viewsheds.npy')
    #                     index = np.loadtxt(f'{folder}/index.csv')
    #                     buildings = index[5]
    #                     vss.append(vs)
    #                 except FileNotFoundError:
    #                     vss.append([])
    #             vsss.append(vss)
    #         self.viewsheds.append(vsss)

    def analyize(self):
        # self.load_data()
        self.correlated_failures()
        self.uncorrelated_failures()

    def uncorrelated_failures(self):
        samples: int = 20
        runs: int = 10
        xs = np.arange(0, 1, 1/samples)
        coverages: np.ndarray = np.zeros(shape=(len(self.comuni)*len(self.sub_area_id),
                                                len(xs),
                                                len(self.rt),
                                                len(self.k),
                                                runs,
                                                len(self.ncovs)+1))
        n = len(self.comuni)*len(self.sub_area_id) * \
            len(xs) * 4 * runs
        pbar = tqdm(total=n)
        for vdx, (area, sa_id) in enumerate(itertools.product(self.comuni, self.sub_area_id)):
            for rtdx, rt in enumerate(self.rt):
                for kdx, k in enumerate(self.k):
                    if k == 3 or rt == 'r1':
                        folder = f'{self.base_dir}/{area}/{self.strategy}/{sa_id}/{rt}/{k}/{self.ratio}/{self.dens}/'
                        v = np.load(f'{folder}/viewsheds.npy')
                        points: range = range(v.shape[0])
                        for rdx, fail_r in enumerate(xs):
                            n_fail: int = int(fail_r * len(points))
                            for rundx in range(runs):
                                local_v: np.ndarray = v.copy()
                                failing_p: list = random.choices(
                                    points, k=n_fail)
                                for p in failing_p:
                                    local_v[p, :, :] = np.zeros_like(
                                        local_v[p])
                                coverages[vdx, rdx, rtdx, kdx,
                                          rundx] = self.calc_coverage(local_v)
                                pbar.update(1)

        for ndx, ncov in enumerate(self.ncovs):
            for rtdx, rt in enumerate(self.rt):
                for kdx, k in enumerate(self.k):
                    if k == 3 or rt == 'r1':
                        rel_cov = coverages[:, :, rtdx, kdx, :,
                                            ndx+1]/coverages[:, :, rtdx, kdx, :, 0]
                        cov_m = rel_cov.mean(axis=(0, 2))
                        cov_e = conf_int(rel_cov, axis=(0, 2))
                        plt.errorbar(xs, cov_m, cov_e, label=f'{rt} k={k}')
            plt.legend()
            plt.ylabel('coverage')
            plt.xlabel('failure ratio')
            plt.savefig(f'{self.base_dir}/reliability_uncorr_{ncov}.pdf')
            plt.clf()

    def correlated_failures(self):
        samples: int = 20
        runs: int = 10
        xs = np.arange(0, 1, 1/samples)
        coverages: np.ndarray = np.zeros(shape=(len(self.comuni)*len(self.sub_area_id),
                                                len(xs),
                                                len(self.rt),
                                                len(self.k),
                                                runs,
                                                len(self.ncovs)+1))
        n = len(self.comuni)*len(self.sub_area_id) * \
            len(xs) * 4 * runs
        pbar = tqdm(total=n)
        for vdx, (area, sa_id) in enumerate(itertools.product(self.comuni, self.sub_area_id)):
            for rtdx, rt in enumerate(self.rt):
                for kdx, k in enumerate(self.k):
                    if k == 3 or rt == 'r1':
                        folder = f'{self.base_dir}/{area}/{self.strategy}/{sa_id}/{rt}/{k}/{self.ratio}/{self.dens}/'
                        v = np.load(f'{folder}/viewsheds.npy')
                        buildings = np.loadtxt(f'{folder}/index.csv')[:, 5]
                        build_dict = {}
                        for idx, b in enumerate(buildings):
                            b = int(b)
                            build_dict[b] = build_dict.get(b, []) + [idx]

                        for rdx, fail_r in enumerate(xs):
                            n_fail: int = int(fail_r * len(buildings))
                            for rundx in range(runs):
                                local_v: np.ndarray = v.copy()
                                failing_b: list = random.choices(buildings,
                                                                 k=n_fail)
                                for b in failing_b:
                                    for p in build_dict[b]:
                                        local_v[p, :, :] = np.zeros_like(
                                            local_v[p])
                                coverages[vdx, rdx, rtdx, kdx, rundx] = self.calc_coverage(
                                    local_v)
                                pbar.update(1)

        for ndx, ncov in enumerate(self.ncovs):
            for rtdx, rt in enumerate(self.rt):
                for kdx, k in enumerate(self.k):
                    if k == 3 or rt == 'r1':
                        rel_cov = coverages[:, :, rtdx, kdx, :,
                                            ndx+1]/coverages[:, :, rtdx, kdx, :, 0]
                        cov_m = rel_cov.mean(axis=(0, 2))
                        cov_e = conf_int(rel_cov, axis=(0, 2))
                        plt.errorbar(xs, cov_m, cov_e, label=f'{rt} k={k}')
            plt.legend()
            plt.ylabel('coverage')
            plt.xlabel('failure ratio')
            plt.savefig(f'{self.base_dir}/reliability_corr_{ncov}.pdf')
            plt.clf()

    def calc_coverage(self, viewshed: 'np.ndarray'):
        coverage: np.ndarray = viewshed.sum(axis=0).flatten()
        areas = np.zeros(shape=(len(self.ncovs)+1))
        areas[0] = len(coverage[coverage >= 0])
        for ndx, ncov in enumerate(self.ncovs):
            areas[ndx+1] = len(coverage[coverage >= ncov])
        return areas


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(
        default_config_files=['bs_analyize.yaml'])

    parser.add_argument(
        "-d", "--dir", help="dir with datafiles", required=True)
    parser.add_argument("-c", "--comune", required=True, action='append')
    parser.add_argument("--dens", type=int, required=True)
    parser.add_argument("--k", type=int, required=True, action='append')
    parser.add_argument("--ranking_type", type=str, action='append')
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument('--sub_area', required=True,
                        type=str, action='append')
    parser.add_argument("--c_ant", type=int, required=True)
    parser.add_argument("--c_build", type=int, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--cached", action='store_true')
    parser.add_argument("--ncovs", type=int,
                        required=True, action='append')

    args = parser.parse_args()
    tn = Reliability(args.dir, args.comune, args.sub_area, args.ratio, args.dens,
                     args.c_ant, args.c_build, args.k, args.ranking_type, args.ncovs, args.strategy, args.cached)
    # tn.analyize_antennaperbs()
    tn.analyize()
    # tn.analyze_gnuplot()
