import configargparse
import numpy as np
from tqdm import tqdm
import scipy.stats
import itertools

import matplotlib.pyplot as plt
from cycler import cycler
import random
from multiprocessing import Pool

max_antper_bs = 60
antper_bs_cost = 5


def conf_int(data, axis=0, alpha=0.95):
    n = 1
    if type(axis) == int:
        n = len(data[axis])
    else:
        for ax in axis:
            n *= data.shape[ax]
    sem = np.std(data, axis=axis, ddof=1) / np.sqrt(n)
    return sem * scipy.stats.t.ppf((1 + alpha) / 2., n-1)


class Reliability():
    def __init__(self, base_dir, comuni, sub_area_id, ratio, denss, c_ant, c_build, k, rt, ncovs, strategy, cached, nruns):
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
        self.nruns = nruns
        self.ncovs = ncovs
        self.strategy = strategy
        self.viewsheds = []
        self.default_cycler = (cycler(color=['r', 'g', 'b', 'y', 'black', 'orange']) +
                               cycler(linestyle=['-', '--', '-', '--', '-', '--']))

    def compute_reliability(self, params: dict):
        local_v = params['local_v'].copy()
        failing_p = params['failing_p']
        for p in failing_p:
            local_v[p] = np.zeros_like(
                local_v[p])
        return self.calc_coverage(local_v)

    def calc_coverage(self, viewshed: 'np.ndarray'):
        coverage: np.ndarray = viewshed.sum(axis=0)
        areas = np.zeros(shape=(len(self.ncovs)+1))
        areas[0] = len(coverage[coverage >= 0])
        for ndx, ncov in enumerate(self.ncovs):
            areas[ndx+1] = len(coverage[coverage >= ncov])
        return areas

    def simulte_failures(self, correlated=False, ranked=False):
        samples: int = 30
        xs = np.linspace(0, 1, samples, endpoint=True)
        corr_str = 'corr' if correlated else 'uncorr'
        ranked_str = 'rank' if ranked else 'random'
        if ranked:
            self.nruns = 1
        try:
            if self.cached:
                compressed = np.load(
                    f'{self.base_dir}/reliability_{corr_str}_{ranked_str}.npz')
                coverages = compressed['arr_0']
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            coverages: np.ndarray = np.zeros(shape=(len(self.comuni)*len(self.sub_area_id),
                                                    len(xs),
                                                    len(self.rt),
                                                    len(self.k),
                                                    self.nruns,
                                                    len(self.ncovs)+1))
            n = len(self.comuni)*len(self.sub_area_id) * len(xs) * 4
            pbar = tqdm(total=n)
            pool = Pool(1)
            for vdx, (area, sa_id) in enumerate(itertools.product(self.comuni, self.sub_area_id)):
                for rtdx, rt in enumerate(self.rt):
                    for kdx, k in enumerate(self.k):
                        if k == 3 or rt == 'r1':
                            folder = f'{self.base_dir}/{area}/{self.strategy}/{sa_id}/{rt}/{k}/{self.ratio}/{self.dens}/'
                            v = np.load(f'{folder}/viewsheds.npy').T
                            buildings = np.loadtxt(f'{folder}/index.csv')[:, 5]
                            indexes = list(
                                map(int, np.loadtxt(f'{folder}/index.csv')[:, 6]))
                            v = v[indexes]
                            all_b_ranked = np.loadtxt(
                                f'{self.base_dir}/{area}/{self.strategy}/{sa_id}/{rt}/{k}/ranking_5.txt', delimiter=',', dtype=int)
                            ranking = []
                            for b in all_b_ranked:
                                if b in buildings:
                                    ranking.append(b)
                            build_dict = {}
                            for idx, b in enumerate(buildings):
                                b = int(b)
                                build_dict[b] = build_dict.get(b, []) + [idx]
                            for rdx, fail_r in enumerate(xs):
                                for rundx in range(self.nruns):
                                    failing_p = self.get_failing_p(
                                        correlated, ranked, v, buildings, build_dict, fail_r, ranking)
                                    rel = self.compute_reliability(
                                        {'local_v': v, 'failing_p': failing_p})
                                    coverages[vdx, rdx, rtdx, kdx, rundx] = rel
                                pbar.update(1)

            np.savez_compressed(
                f'{self.base_dir}/reliability_{corr_str}_{ranked_str}', coverages)
        for ndx, ncov in enumerate(self.ncovs):
            cov_gnuplot = np.zeros(
                shape=(len(xs), 4*2+1))
            cov_gnuplot[:, 0] = xs
            i = 0
            for rtdx, rt in enumerate(self.rt):
                for kdx, k in enumerate(self.k):
                    if k == 3 or rt == 'r1':
                        rel_cov = coverages[:, :, rtdx, kdx, :,
                                            ndx+1]/coverages[:, :, rtdx, kdx, :, 0]
                        cov_m = rel_cov.mean(axis=(0, 2))
                        cov_e = conf_int(rel_cov, axis=(0, 2))
                        cov_gnuplot[:, i+1] = cov_m
                        cov_gnuplot[:, i+5] = cov_e
                        plt.errorbar(xs*100, cov_m, cov_e, label=f'{rt} k={k}')
                        i += 1
            plt.legend()
            plt.ylabel('coverage')
            plt.xlabel('failure ratio (%)')

            plt.savefig(
                f'{self.base_dir}/reliability_{corr_str}_{ranked_str}_{ncov}.pdf')
            plt.clf()
            np.savetxt(
                f'{self.base_dir}/reliability_{corr_str}_{ranked_str}_{ncov}.csv', cov_gnuplot)

    def get_failing_p(self, correlated, ranked, v, buildings, build_dict, fail_r, ranking):
        if correlated:
            n_fail: int = int(fail_r * len(set(buildings)))
            if ranked:
                # ranking = v.sum(axis=1)
                # dict = {build: ranking[points].sum()
                #         for build, points in build_dict.items()}
                # sorted_d = [d[0]
                #             for d in sorted(dict.items(), key=lambda x: x[1], reverse=True)]
                failing_b = ranking[:n_fail]

            else:
                failing_b: list = random.sample(set(buildings), k=n_fail)
            failing_p = []
            for b in failing_b:
                for p in build_dict[b]:
                    failing_p.append(p)
        else:
            n_fail: int = int(fail_r * v.shape[0])
            if ranked:
                ranking = v.sum(axis=1).argsort()[::-1]
                failing_p = ranking[:n_fail]
            else:
                failing_p: list = random.sample(range(v.shape[0]), k=n_fail)
        return failing_p

    def analyize(self):
        # self.simulte_failures(correlated=False, ranked=False)
        # self.simulte_failures(correlated=False, ranked=True)
        # self.simulte_failures(correlated=True, ranked=False)
        self.simulte_failures(correlated=True, ranked=True)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(
        default_config_files=['reliability.yaml'])

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
    parser.add_argument("--nruns", type=int, required=True)

    args = parser.parse_args()
    tn = Reliability(args.dir, args.comune, args.sub_area, args.ratio, args.dens,
                     args.c_ant, args.c_build, args.k, args.ranking_type, args.ncovs, args.strategy, args.cached, args.nruns)
    # tn.analyize_antennaperbs()
    tn.analyize()
    # tn.analyze_gnuplot()
