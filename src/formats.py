from os.path import join
from tempfile import TemporaryDirectory
from spikeinterface.core import BaseSorting
from spikeinterface.exporters import export_to_phy as to_phy
from spikeinterface.core.waveform_extractor import WaveformExtractor
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv, DataFrame as df
from abc import ABC, abstractmethod
from itertools import accumulate


class SpikeFormat(ABC):
    @abstractmethod
    def save(self, path):
        return


SUPPORTED_FORMATS = ["phy", "clu-res"]


class Phy(SpikeFormat):
    def __init__(self, recording, sorting):
        """ Convert sorted data to phy format.
            Args:
                recording - recording extractor (has to be filtered)
                sorting - sorting extractor
        """
        self.wfe_cache_dir = TemporaryDirectory()
        self.wfe = WaveformExtractor.create(recording, sorting,
                   self.wfe_cache_dir.name, remove_if_exists=True)
        self.wfe.set_params()
        self.wfe.run_extract_waveforms()

    def __del__(self):
        self.wfe_cache_dir.cleanup()

    def save(self, out_dir):
        """ Save sorted spikes in phy format to the given `out_dir`. """
        sw.plot_unit_waveforms(self.wfe)
        plt.savefig(join(out_dir, "waveforms.png"))
        to_phy(self.wfe, join(out_dir, "phy_export")) #, copy_binary=False)


class CluRes(SpikeFormat):
    """ Convert sorted data to clu-res format.
        When converting from phy ignore "unsorted" clusters.
    """
    def __init__(self, clu, res):
        assert len(clu) == len(res)
        self.clu = clu
        self.res = res
        self.num_clusters = max(clu) - 1 if len(clu) > 0 else 0

    def save(self, base_path):
        """ Save .clu and .res to the given base_path (append extensions). """
        np.savetxt(base_path + ".clu", self.clu, fmt="%i")
        np.savetxt(base_path + ".res", self.res, fmt="%i")

    @classmethod
    def load(cls, base_path):
        """ Load .clu and .res from the given base_path (append extensions). """
        clu = np.loadtxt(base_path + ".clu", dtype=int)
        res = np.loadtxt(base_path + ".res", dtype=int)
        return CluRes(clu, res)

    @classmethod
    def merge_tetrodes(cls, tets: dict):
        """ Merge clusters from several tetrodes into one CluRes.

            Args:
                tets - dict {tet: CluRes}
            Return:
                Merged, CluRes, origins (dict {cluster: tet})
        """
        cluress = {tet: cr for tet, cr in tets.items() if cr.num_clusters > 0}
        clu_shifts = accumulate([0] + [cr.num_clusters for cr in cluress.values()][:-1])
        clu_shifts = dict(zip(cluress.keys(), clu_shifts))

        clu_merged, res_merged, origins = [], [], {}
        for tet, cs in clu_shifts.items():
            res_merged.extend(cluress[tet].res)
            clu = cluress[tet].clu.copy()
            clu[clu > 1] += cs  # 0-noise, 1-mua so we keep them the same from every tet
            clu_merged.extend(clu)
            for c in np.unique(clu):
                origins[c] = tet
            origins[0] = origins[1] = '-'

        srted = sorted(zip(res_merged, clu_merged))
        res_merged, clu_merged = zip(*srted)
        return CluRes(clu_merged, res_merged), origins

    @classmethod
    def from_sorting(cls, sorting: BaseSorting):
        """ Extract .clu and .res from a sorting extractor.
            Args:
                sorting - instance of spikeinterface.core.BaseSorting
            Return:
                A CluRes object.
        """
        uids = sorting.unit_ids
        cluster_groups = df({'cluster_id': [i for i in range(len(uids))],
                             'group': ['autosorted'] * len(uids)})
        all_spikes = sorting.get_all_spike_trains(outputs='unit_index')
        spike_times = all_spikes[0][0][:, np.newaxis]
        spike_labels = all_spikes[0][1][:, np.newaxis]
        return CluRes.__from_spikes(cluster_groups, spike_times, spike_labels)

    @classmethod
    def from_phy(cls, phy_dir):
        """ Extract .clu and .res from data saved in phy format. Ignore "unsorted" clusters.
            Args:
                phy_dir - path to directory with data in phy format
            Return:
                A CluRes object.
        """
        cluster_groups = read_csv(join(phy_dir, "cluster_group.tsv"), sep="\t")
        spike_times = np.load(join(phy_dir, "spike_times.npy"))
        spike_clusters = np.load(join(phy_dir, "spike_clusters.npy"))
        return CluRes.__from_spikes(cluster_groups, spike_times, spike_clusters)

    @classmethod
    def __from_spikes(cls, cluster_groups, spike_times, spike_clusters):
        # remove spikes from unsorted clusters
        unsorted_clusters = cluster_groups[cluster_groups["group"] == "unsorted"]["cluster_id"].tolist()
        for usc in unsorted_clusters:
            inds = spike_clusters == usc
            spike_clusters[inds] = -1
            spike_times[inds] = -1
        spike_clusters = spike_clusters[spike_clusters != -1]
        res = spike_times[spike_times != -1]
        clu = spike_clusters.copy()

        # mark all noise spikes with 0
        noise_clusters = cluster_groups[cluster_groups["group"] == "noise"]["cluster_id"].tolist()
        for c in noise_clusters:
            clu[spike_clusters == c] = 0

        # mark all mua spikes with 1
        mua_clusters = cluster_groups[cluster_groups["group"] == "mua"]["cluster_id"].tolist()
        for c in mua_clusters:
            clu[spike_clusters == c] = 1

        # good clusters are from 2 onwards
        good_clusters = cluster_groups[cluster_groups["group"] == "good"]["cluster_id"].tolist()
        for i, c in enumerate(good_clusters):
            clu[spike_clusters == c] = i + 2

        assert len(clu) == len(res)
        return CluRes(clu, res)
