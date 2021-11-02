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

max_antper_bs = 60
antper_bs_cost = 5


def conf_int(data, axis=0, alpha=0.95):
    return scipy.stats.sem(data, axis=axis) * scipy.stats.t.ppf((1 + alpha) / 2., data.shape[axis]-1)


class BS_Analysis():
    def __init__(self, base_dir, comuni, sub_area_id, ratio, denss, c_ant, c_build, ncovs, k, rt, strategy, cached):
        self.srid = 3003  # TODO: check if it is metric
        self.crs = "EPSG:%4d" % (self.srid)
        self.base_dir = base_dir
        self.denss = denss
        self.comuni = comuni
        self.ratio = ratio
        self.sub_area_id = sub_area_id
        self.c_ant = c_ant
        self.c_build = c_build
        self.ncovs = ncovs
        self.cached = cached
        self.k = k
        self.rt = rt
        self.strategy = strategy
        self.default_cycler = (cycler(color=['r', 'g', 'b', 'y', 'black', 'orange']) +
                               cycler(linestyle=['-', '--', '-', '--', '-', '--']))

    def load_data(self):
        try:
            if(self.cached):
                self.scalars = np.load(
                    f'{self.base_dir}/{self.strategy}_scalars.npy')
                self.areas = np.load(
                    f'{self.base_dir}/{self.strategy}_areas.npy')
                self.dists = np.load(
                    f'{self.base_dir}/{self.strategy}_dists.npy')
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            self.areas = np.zeros((len(self.denss), len(self.ratio), len(
                self.sub_area_id)*len(self.comuni), len(self.rt), len(self.k), len(self.ncovs)+1))
            self.scalars = np.zeros(shape=(len(self.denss), len(self.ratio), len(
                self.sub_area_id)*len(self.comuni), len(self.rt), len(self.k), 4))
            self.dists = np.zeros((len(self.denss), len(self.ratio), len(
                self.sub_area_id)*len(self.comuni), len(self.rt), len(self.k), max_antper_bs-1))
            for rdx, r in enumerate(self.rt):
                for kdx, kcov in enumerate(self.k):
                    for (i, ratio), (j, dens) in tqdm(itertools.product(enumerate(self.ratio), enumerate(self.denss)), total=len(self.ratio)*len(self.denss)):
                        for k, (area, sa_id) in enumerate(itertools.product(self.comuni, self.sub_area_id)):
                            folder = f'{self.base_dir}/{area}/{self.strategy}/{sa_id}/{r}/{kcov}/{ratio}/{dens}/'
                            files_dist = glob.glob(f'{folder}/*_uncropped.tif')
                            try:
                                coverage = np.loadtxt(
                                    files_dist[0].replace('_uncropped.tif', '.csv'))
                                self.scalars[j, i, k, rdx, kdx] = np.loadtxt(
                                    f'{folder}/metrics.csv')
                                index = np.loadtxt(f'{folder}/index.csv')
                                occurencies = list(
                                    Counter(index[:, 5]).values())
                                self.dists[j, i, k, rdx, kdx] = np.histogram(
                                    occurencies, bins=max_antper_bs-1, range=(1, max_antper_bs))[0]

                            except IndexError:
                                print(f'{folder}/*_uncropped.tif not found')
                                continue
                            self.areas[j, i, k, rdx, kdx, 0] = len(
                                coverage[coverage >= 0])
                            for ndx, ncov in enumerate(self.ncovs):
                                self.areas[j, i, k, rdx, kdx, ndx +
                                           1] = len(coverage[coverage >= ncov])

            np.save(f'{self.base_dir}/{self.strategy}_scalars.npy', self.scalars)
            np.save(f'{self.base_dir}/{self.strategy}_areas.npy', self.areas)
            np.save(f'{self.base_dir}/{self.strategy}_dists.npy', self.dists)

    def calc_cost_funct(self, type=0):
        # Create a cost vector [1,1,1,1,1,2,2,2,2,2] etc
        cost_vector = np.ceil(
            np.array(range(1, self.dists.shape[5]+1))/antper_bs_cost)
        # Expand it to all the dimensions
        cost_matrix = np.array([[[[[cost_vector]*len(self.k)]*len(self.rt)]*len(
            self.comuni)*len(self.sub_area_id)]*len(self.ratio)]*len(self.denss))

        if type == 0:
            self.costs = (self.c_ant * self.scalars[:, :, :, :, :, 3] +
                          self.c_build * self.scalars[:, :, :, :, :, 2]) * 1e-6  # In Millions
        elif type == 1:
            # the building cost is the sum of the building distribution (number of antenna for each build) times the cost of the build times the matrix cost:
            # Eg:
            # dist = 10,5,4,4,3,2,2,1,0,0
            # cost_matrix = 1,1,1,1,1,2,2,2,2,2
            # c_build = 1000
            # cost_buildings = [10*1 + 5*1 + 5*1 + 4*1 + 3*1 + 2*2 + 2*2 + 1*2] * 1000
            cost_buildings = (
                self.dists * cost_matrix).sum(axis=5) * self.c_build
            cost_ant = self.c_ant * self.scalars[:, :, :, :, :, 3]
            self.costs = 1e-6 * (cost_buildings + cost_ant)  # In Millions

    def relative_coverage_gnuplot_wcnc(self):
        coverages = np.zeros((len(self.denss), len(self.ratio), len(
            self.sub_area_id)*len(self.comuni), len(self.rt), len(self.k), len(self.ncovs)))
        for ndx in range(len(self.ncovs)):
            coverages[:, :, :, :, :, ndx] = self.areas[:, :,
                                                       :, :, :, ndx+1] / self.areas[:, :, :, :, :, 0]
        # np.zeros((len(self.denss), len(self.ratio), len(
        #     self.sub_area_id)*len(self.comuni), len(self.rt), len(self.k), len(self.ncovs)+1))
        coverage_gnuplot = np.zeros(shape=(len(self.denss), len(self.ratio)*2+1,
                                    len(self.ncovs)))
        coverage_gnuplot[:, 0] = np.array([self.denss]*(len(self.ncovs))).T
        coverage_gnuplot[:, 1:len(self.ratio)+1,
                         :] = coverages.mean(axis=2)[:, :, 0, 0, :]
        coverage_gnuplot[:, len(self.ratio)+1:,
                         :] = conf_int(coverages, axis=2)[:, :, 0, 0, :]
        for ndx, ncov in enumerate(self.ncovs):
            np.savetxt(f'{self.base_dir}/wcnc/urban_coverage_{ncov}.csv',
                       coverage_gnuplot[:, :, ndx])

    def analyize_antennaperbs(self):
        meaned_d = self.dists.sum(axis=2)
        plt.figure(figsize=(5*len(self.denss), 5*len(self.ratio)))
        k = 0
        for (i, ratio), (j, dens) in tqdm(itertools.product(enumerate(self.ratio), enumerate(self.denss)), total=len(self.ratio)*len(self.denss)):
            ax = plt.subplot(len(self.ratio), len(self.denss), k+1)
            ax.bar(range(1, max_antper_bs), meaned_d[j, i])
            ax.set_title(
                f'buildings ={self.scalars[j,i,:,1].mean():.2f} ({ratio}%) , gNB={self.scalars[j,i,:,3].mean():.2f} ({dens})')
            # ax.set_xticklabels(self.denss)
            k += 1
        plt.tight_layout()
        plt.savefig(f'{self.base_dir}/building_ant_dist.pdf')

    def calc_build_ratio(self):
        # Build ratio
        build_ratio = self.scalars[:, :, :, :,
                                   :, 2]/self.scalars[:, :, :, :, :, 0]

        ratio_avg = build_ratio.mean(axis=2)
        ratio_err = conf_int(build_ratio, axis=2)
        plt.figure(figsize=(len(self.ratio)*5, 1*5))

        for rtdx, ratio in enumerate(self.ratio):
            ax = plt.subplot(1, len(self.ratio), rtdx+1)
            ax.set_prop_cycle(self.default_cycler)
            ax.set_xlabel("gNB/kmq")
            for rdx, rt in enumerate(self.rt):
                for kdx, k in enumerate(self.k):
                    if k == 3 or rt == 'r1':
                        ax.errorbar(
                            self.denss, ratio_avg[:, rtdx, rdx, kdx], ratio_err[:, rtdx, rdx, kdx], label=f'{rt}, k={k}')
            ax.set_title(f"{ratio}% building")
            ax.set_ylabel(f"building ratio")
            ax.legend()
        plt.suptitle(f"Building ratio ({self.strategy})")
        plt.savefig(f'{self.base_dir}/{self.strategy}_ratio.pdf')
        plt.clf()

        # build_ratio_gnuplot = np.zeros(shape=(len(self.denss), len(self.ratio)*2+1))
        # build_ratio_gnuplot[:, 0] = self.denss

        # build_ratio_gnuplot[:, 1:len(self.ratio)+1] = build_ratio.mean(axis=2)
        # build_ratio_gnuplot[:, len(self.ratio)+1:] = conf_int(build_ratio, axis=2)
        # np.savetxt(f'{self.base_dir}/urban_buildrat.csv', build_ratio_gnuplot)

    def calc_cost(self):
        costs_m = self.costs.mean(axis=2)
        costs_e = conf_int(self.costs, axis=2)
        plt.figure(figsize=(len(self.ratio)*5, 1*5))
        for rtdx, ratio in enumerate(self.ratio):
            ax = plt.subplot(1, len(self.ratio), rtdx+1)
            ax.set_prop_cycle(self.default_cycler)
            ax.set_xlabel("gNB/kmq")
            # if ratio==10:
            #     ax.set_ylim(0,1)
            # elif ratio== 100:
            #     ax.set_ylim(0,2)
            for rdx, rt in enumerate(self.rt):
                for kdx, k in enumerate(self.k):
                    if k == 3 or rt == 'r1':
                        ax.errorbar(
                            self.denss, costs_m[:, rtdx, rdx, kdx], costs_e[:, rtdx, rdx, kdx], label=f'{rt}, k={k}')

            n_bs = self.scalars[:, 0, :, rdx, kdx, 3].mean(axis=1)
            err = conf_int(self.scalars[:, 0, :, rdx, kdx, 3], axis=1)

            ax.plot(self.denss, 1e-6 * (self.c_build + self.c_ant)
                    * (n_bs + err), label='Upper Bound')
            ax.plot(self.denss, 1e-6 * (self.c_ant * (n_bs - err) +
                    self.c_build), label='Lower Bound')
            ax.set_title(f"{ratio}% building")
            ax.set_ylabel(f"Million $")
            ax.legend()
        plt.suptitle(f"Cost (M$) ({self.strategy})")
        plt.savefig(f'{self.base_dir}/{self.strategy}_cost.pdf')
        plt.clf()

        # #Cost
        # cost_gnuplot = np.zeros(shape=(len(self.denss), len(self.ratio)*2+3))
        # cost_gnuplot[:, 0] = self.denss
        # costs = self.costs
        # cost_gnuplot[:, 1:len(self.ratio)+1] = costs.mean(axis=2)
        # cost_gnuplot[:, len(self.ratio)+1:-2] = conf_int(costs, axis=2)
        # ##Upper and lower bound
        # n_bs = self.scalars[:,0,:,3].mean(axis=1)
        # err = conf_int(self.scalars[:,0,:,3], axis=1)
        # cost_gnuplot[:, -2] = 1e-6 * (self.c_build + self.c_ant) * (n_bs + err)
        # cost_gnuplot[:, -1] = 1e-6 * (self.c_ant * (n_bs - err) + self.c_build)
        # np.savetxt(f'{self.base_dir}/urban_cost.csv', cost_gnuplot)

    def calc_norm_coverage(self):
        # Normalized coverage
        coverages = np.zeros((len(self.denss), len(self.ratio), len(
            self.sub_area_id)*len(self.comuni), len(self.rt), len(self.k), len(self.ncovs)))
        for ndx in range(len(self.ncovs)):
            coverages[:, :, :, :, :, ndx] = self.areas[:, :,
                                                       :, :, :, ndx+1] / self.areas[:, :, :, :, :, 0]
        avg_cov = coverages.mean(axis=2)
        err_cov = conf_int(coverages, axis=2)
        i = 0
        plt.figure(figsize=(len(self.ratio)*5, len(self.ncovs)*5))
        for ndx, ncov in enumerate(self.ncovs):
            for rtdx, ratio in enumerate(self.ratio):
                ax = plt.subplot(len(self.ncovs), len(self.ratio), i+1)
                ax.set_prop_cycle(self.default_cycler)
                ax.set_xlabel("gNB/kmq")
                i += 1
                for rdx, rt in enumerate(self.rt):
                    for kdx, k in enumerate(self.k):
                        if k == 3 or rt == 'r1':
                            ax.errorbar(
                                self.denss, avg_cov[:, rtdx, rdx, kdx, ndx], err_cov[:, rtdx, rdx, kdx, ndx], label=f'{rt}, k={k}')
                ax.set_ylim(0.1, 1)
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                ax.set_yticks(np.arange(0.1, 1.1, step=0.1))
                ax.grid(which='both')
                ax.set_title(f"{ncov}-coverage, {ratio}% building")
                ax.set_ylabel(f"c_{ncov}")
                ax.legend()
        plt.suptitle(f"Relative coverage ({self.strategy})")
        plt.savefig(f'{self.base_dir}/{self.strategy}_coverage.pdf')
        plt.clf()

    def calc_marginal_cost(self):
        # Marginal cost on marginal coverage
        meaned_areas = self.areas.mean(axis=2)
        inc_coverage = np.zeros_like(meaned_areas)
        inc_coverage[0] = meaned_areas[0]
        inc_coverage[1:] = meaned_areas[1:] - meaned_areas[:-1]
        costs = (self.c_ant * self.scalars[:, :, :, :, :, 3] +
                 self.c_build * self.scalars[:, :, :, :, :, 2]) * 1e-6  # In Millions
        meaned_cost = costs.mean(axis=2)
        inc_cost = np.zeros_like(meaned_cost)
        inc_cost[0] = meaned_cost[0]
        inc_cost[1:] = meaned_cost[1:] - meaned_cost[:-1]

        for ndx, ncov in enumerate(self.ncovs):
            plt.figure(figsize=(5*len(self.ratio), 5))
            l = 1
            cost_inc_mq = inc_cost[:, :, :, :] / \
                inc_coverage[:, :, :, :, ndx+1]
            for rtdx, ratio in enumerate(self.ratio):
                ax = plt.subplot(1, len(self.ratio), l)
                for rdx, rt in enumerate(self.rt):
                    for kdx, k in enumerate(self.k):
                        if k == 3 or rt == 'r1':
                            ax.plot(
                                self.denss, cost_inc_mq[:, rtdx, rdx, kdx], label=f'{rt}, k={k}')
                            ax.legend()
                            ax.set_title(f'{ratio}%')
                l = l+1
            plt.suptitle(f'Marginal {ncov}-coverage')
            plt.savefig(
                f'{self.base_dir}/{self.strategy}_costcoverage_{ncov}.pdf')
            plt.clf()

            #out = np.zeros(shape=(len(self.denss), len(self.ratio)+1))
            # out[:, 1:] = cost_inc_mq
            # out[:, 0] = np.array(self.denss)
            #np.savetxt(f'{self.base_dir}/urban_costcoverage_{ncov}.csv', out)

    def calc_scatter_costcoverage(self):
        coverages = np.zeros((len(self.denss), len(self.ratio), len(
            self.sub_area_id)*len(self.comuni), len(self.ncovs)-1))
        for ncov in self.ncovs[1:]:
            coverages[:, :, :, ncov-1] = self.areas[:,
                                                    :, :, ncov] / self.areas[:, :, :, 0]
        costs = self.costs
        scatters = np.zeros(shape=(len(self.ratio), len(
            self.denss)*len(self.comuni)*len(self.sub_area_id), 2))
        for i, ratio in enumerate(self.ratio):
            l = 0
            for j, dens in enumerate(self.denss):
                for k, (area, sa_id) in enumerate(itertools.product(self.comuni, self.sub_area_id)):
                    scatters[i, l, 0] = coverages[j, i, k, 0]
                    scatters[i, l, 1] = costs[j, i, k]
                    l += 1
            plt.scatter(scatters[i, :, 0], scatters[i, :, 1], label=f"{ratio}")
        plt.legend()
        plt.xlabel("coverage")
        plt.ylabel("cost (M£)")
        plt.savefig(f'{self.base_dir}/scatter.pdf')
        plt.clf()

    def calc_scatter_costcoverage_avg(self):
        #gnuplot_out = np.zeros(shape=(len(self.denss), len(self.ratio)*4))
        coverages = np.zeros((len(self.denss), len(self.ratio), len(
            self.sub_area_id)*len(self.comuni), len(self.rt), len(self.k), len(self.ncovs)))
        for ndx in range(len(self.ncovs)):
            coverages[:, :, :, :, :, ndx] = self.areas[:, :,
                                                       :, :, :, ndx+1] / self.areas[:, :, :, :, :, 0]
        costs = self.costs
        scatters = np.zeros(shape=(len(self.ratio), len(
            self.denss), len(self.rt), len(self.k), 4))

        for ndx, ncov in enumerate(self.ncovs):
            for i, ratio in enumerate(self.ratio):
                for j, dens in enumerate(self.denss):
                    for rdx, rt in enumerate(self.rt):
                        for kdx, k in enumerate(self.k):
                            scatters[i, j, rdx, kdx, 0] = coverages[j,
                                                                    i, :, rdx, kdx, ndx].mean()
                            scatters[i, j, rdx, kdx, 1] = costs[j,
                                                                i, :, rdx, kdx].mean()
                            scatters[i, j, rdx, kdx, 2] = conf_int(
                                coverages[j, i, :, rdx, kdx, ndx])
                            scatters[i, j, rdx, kdx, 3] = conf_int(
                                costs[j, i, :, rdx, kdx])

            plt.figure(figsize=(5*len(self.ratio), 5))
            l = 1
            for i, ratio in enumerate(self.ratio):
                ax = plt.subplot(1, len(self.ratio), l)
                for rdx, rt in enumerate(self.rt):
                    for kdx, k in enumerate(self.k):
                        if k == 3 or rt == 'r1':
                            ax.errorbar(scatters[i, :, rdx, kdx, 0], scatters[i, :, rdx, kdx, 1], xerr=scatters[i,
                                        :, rdx, kdx, 2], yerr=scatters[i, :, rdx, kdx, 3], fmt='o', label=f"k={k} {rt}")
                # ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                ax.xaxis.set_minor_locator(MultipleLocator(0.05))
                #ax.set_yticks(np.arange(0.1, 1.1, step=0.1))
                ax.grid(which='both')
                ax.legend()
                ax.set_title(f"{ratio} %")
                l = l+1
            plt.savefig(
                f'{self.base_dir}/{self.strategy}_scatter_mean_{ncov}.pdf')
        #     for k in range(4):
        #         gnuplot_out[:, 4*i+k] = scatters[i,:,k]
        # np.savetxt(f'{self.base_dir}/scatter_mean.csv', gnuplot_out)

    def analyize_pyplot(self):
        self.load_data()
        self.calc_cost_funct(0)
        self.calc_build_ratio()
        self.calc_cost()
        self.calc_norm_coverage()
        self.calc_marginal_cost()
        # self.calc_scatter_costcoverage()
        self.calc_scatter_costcoverage_avg()
        # self.analyize_antennaperbs()

    def analyze_gnuplot(self):
        self.load_data()
        self.calc_cost_funct(0)
        self.relative_coverage_gnuplot_wcnc()
        self.calc_scatter_costcoverage_avg_gnuplot_wcnc()
        self.calc_cost_gnuplot_wcnc()
        self.calc_marginal_cost_gnuplot_wcnc()

    def calc_scatter_costcoverage_avg_gnuplot_wcnc(self):
        gnuplot_out = np.zeros(shape=(len(self.denss), len(self.ratio)*4))
        coverages = np.zeros((len(self.denss), len(self.ratio), len(
            self.sub_area_id)*len(self.comuni), len(self.rt), len(self.k), len(self.ncovs)))
        for ndx in range(len(self.ncovs)):
            coverages[:, :, :, :, :, ndx] = self.areas[:, :,
                                                       :, :, :, ndx+1] / self.areas[:, :, :, :, :, 0]
        costs = self.costs
        scatters = np.zeros(shape=(len(self.ratio), len(
            self.denss), len(self.rt), len(self.k), len(self.ncovs), 4))

        for ndx, ncov in enumerate(self.ncovs):
            for i, ratio in enumerate(self.ratio):
                for rdx, rt in enumerate(self.rt):
                    for kdx, ks in enumerate(self.k):
                        for j, dens in enumerate(self.denss):
                            scatters[i, j, rdx, kdx, ndx, 0] = \
                                coverages[j, i, :, rdx, kdx, ndx].mean()
                            scatters[i, j, rdx, kdx, ndx, 1] =   \
                                costs[j, i, :, rdx, kdx].mean()
                            scatters[i, j, rdx, kdx, ndx, 2] = \
                                conf_int(coverages[j, i, :, rdx, kdx, ndx])
                            scatters[i, j, rdx, kdx, ndx, 3] = \
                                conf_int(costs[j, i, :, rdx, kdx])
                for k in range(4):
                    gnuplot_out[:, 4*i+k] = scatters[i, :, 0, 0, 0, k]
        np.savetxt(f'{self.base_dir}/wcnc/scatter_mean.csv', gnuplot_out)

    def calc_cost_gnuplot_wcnc(self):
        cost_gnuplot = np.zeros(shape=(len(self.denss), len(self.ratio)*2+3))
        cost_gnuplot[:, 0] = self.denss
        costs = self.costs
        cost_gnuplot[:, 1:len(self.ratio)+1] = costs.mean(axis=2)[:, :, 0, 0]
        cost_gnuplot[:, len(self.ratio)+1:-
                     2] = conf_int(costs, axis=2)[:, :, 0, 0]
        # Upper and lower bound
        n_bs = self.scalars[:, 0, :, :, :, 3].mean(axis=1)[:, 0, 0]
        err = conf_int(self.scalars[:, 0, :, :, :, 3], axis=1)[:, 0, 0]
        cost_gnuplot[:, -2] = 1e-6 * (self.c_build + self.c_ant) * (n_bs + err)
        cost_gnuplot[:, -1] = 1e-6 * (self.c_ant * (n_bs - err) + self.c_build)
        np.savetxt(f'{self.base_dir}/wcnc/urban_cost.csv', cost_gnuplot)

    def calc_marginal_cost_gnuplot_wcnc(self):
        # Marginal cost on marginal coverage
        meaned_areas = self.areas.mean(axis=2)
        inc_coverage = np.zeros_like(meaned_areas)
        inc_coverage[0] = meaned_areas[0]
        inc_coverage[1:] = meaned_areas[1:] - meaned_areas[:-1]
        costs = (self.c_ant * self.scalars[:, :, :, :, :, 3] +
                 self.c_build * self.scalars[:, :, :, :, :, 2]) * 1e-6  # In Millions
        meaned_cost = costs.mean(axis=2)
        inc_cost = np.zeros_like(meaned_cost)
        inc_cost[0] = meaned_cost[0]
        inc_cost[1:] = meaned_cost[1:] - meaned_cost[:-1]

        for ndx, ncov in enumerate(self.ncovs):
            cost_inc_mq = inc_cost[:, :, 0, 0] / \
                inc_coverage[:, :, 0, 0, ndx+1]
            out = np.zeros(shape=(len(self.denss), len(self.ratio)+1))
            out[:, 1:] = cost_inc_mq
            out[:, 0] = np.array(self.denss)
            np.savetxt(
                f'{self.base_dir}/wcnc/urban_costcoverage_{ncov}.csv', out)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(
        default_config_files=['analyize.yaml'])

    parser.add_argument(
        "-d", "--dir", help="dir with datafiles", required=True)
    parser.add_argument("-c", "--comune", required=True, action='append')
    parser.add_argument("--dens", type=int, required=True, action='append')
    parser.add_argument("--ncovs", type=int, required=True, action='append')
    parser.add_argument("--k", type=int, required=True, action='append')
    parser.add_argument("--ranking_type", type=str,
                        required=True, action='append')
    parser.add_argument("--ratio", type=float, required=True,
                        default=[], action='append')
    parser.add_argument('--sub_area', required=True, type=str, action='append')
    parser.add_argument("--c_ant", type=int, required=True)
    parser.add_argument("--c_build", type=int, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--cached", action='store_true')

    args = parser.parse_args()
    tn = BS_Analysis(args.dir, args.comune, args.sub_area, args.ratio, args.dens,
                     args.c_ant, args.c_build, args.ncovs, args.k, args.ranking_type, args.strategy, args.cached)
    # tn.analyize_antennaperbs()
    # tn.analyize_pyplot()
    tn.analyze_gnuplot()
