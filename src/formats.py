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
    def __init__(self, sorting: BaseSorting=None, phy_dir=None):
        """ Convert sorted data to clu-res format.
            When converting from phy ignore "unsorted" clusters.
            Args:
                sorting - sorting extractor (if provided don't pass `phy_dir`)
                phy_dir - path to a folder with data in phy format (if provided don't pass `sorting`)
        """
        if sorting is not None and phy_dir is not None:
            raise ValueError("Provide only `sorting` or only `phy_dir`")
        if sorting is not None:
            self.clu, self.res = CluRes.__from_sorting(sorting)
        if phy_dir is not None:
            self.clu, self.res = CluRes.__from_phy(phy_dir)

    def save(self, path):
        """ Save .clu and .res to the given path (append extensions). """
        print(self.clu.shape, self.res.shape)
        np.savetxt(path + ".clu", self.clu, fmt="%i")
        np.savetxt(path + ".res", self.res, fmt="%i")

    @classmethod
    def __from_sorting(cls, sorting: BaseSorting):
        """ Extract .clu and .res from a sorting extractor.
            Args:
                sorting - instance of spikeinterface.core.BaseSorting
            Return:
                clu, res - (np.ndarray, np.ndarray)
        """
        uids = sorting.unit_ids
        cluster_groups = df({'cluster_id': [i for i in range(len(uids))],
                             'group': ['autosorted'] * len(uids)})
        all_spikes = sorting.get_all_spike_trains(outputs='unit_index')
        spike_times = all_spikes[0][0][:, np.newaxis]
        spike_labels = all_spikes[0][1][:, np.newaxis]
        return CluRes.__from_spikes(cluster_groups, spike_times, spike_labels)

    @classmethod
    def __from_phy(cls, phy_dir):
        """ Extract .clu and .res from data saved in phy format. Ignore "unsorted" clusters.
            Args:
                phy_dir - path to directory with data in phy format
            Return:
                clu, res - (np.ndarray, np.ndarray)
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
        return clu, res
