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
        self.wfe.set_params(dtype=float)
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

        clu-res format is based on JC .clu, .res etc. files.
        Each .clu file contains total number of clusters in the firts line.
        0 - is noise (not counted in the total number)
        1 - mua (counted in the total number)

        For now, this class handles folowing files:
            * .clu
            * .res
            * .des (optionally, if available)
    """
    def __init__(self, clu, res, des=None):
        assert len(clu) == len(res) or len(clu) == len(res) + 1
        if des is not None:
            # print(len(clu), len(des), np.max(clu))
            assert len(clu) == 0 or len(des) == np.max(clu)
        if len(clu) == len(res) + 1:
            assert np.max(clu) == clu[0]
        self._clu = clu
        self.res = res
        self.des = des
        self.num_clusters = np.max(clu) if len(clu) > 0 else 0

    @property
    def clu(self):
        return self._clu if len(self._clu) == len(self.res) else self._clu[1:]

    def save(self, base_path):
        """ Save .clu and .res to the given base_path (append extensions). """
        # we want to save the total number of clusters in the first line
        np.savetxt(base_path + ".clu", self._clu, fmt="%i")
        np.savetxt(base_path + ".res", self.res, fmt="%i")
        if self.des is not None:
            np.savetxt(base_path + ".des", self.des, fmt="%s")

    @classmethod
    def load(cls, base_path):
        """ Load .clu and .res from the given base_path (append extensions). """
        clu = np.loadtxt(base_path + ".clu", dtype=int)
        res = np.loadtxt(base_path + ".res", dtype=int)
        try:
            des = np.loadtxt(base_path + ".des", dtype=str)
        except:
            des = None
        return CluRes(clu, res, des)

    @classmethod
    def merge_tetrodes(cls, tets: dict):
        """ Merge clusters from several tetrodes into one CluRes.

            Args:
                tets - dict {tet: CluRes}
            Return:
                Merged, CluRes, origins (dict {cluster: tet})
        """
        cluress = {tet: cr for tet, cr in tets.items() if cr.num_clusters > 1}
        clu_shifts = accumulate([0] + [cr.num_clusters-1 for cr in cluress.values()][:-1])
        clu_shifts = dict(zip(cluress.keys(), clu_shifts))

        clu_merged, res_merged, origins = [], [], {}
        for tet, cs in clu_shifts.items():
            res_merged.extend(cluress[tet].res)
            clu = np.array(cluress[tet].clu.copy())
            clu[clu > 1] += cs  # 0-noise, 1-mua so we keep them the same from every tet
            clu_merged.extend(clu)
            for c in np.unique(clu):
                origins[c] = tet
            origins[0] = origins[1] = '-'

        srted = sorted(zip(res_merged, clu_merged))  # sort by res_merged
        res_merged, clu_merged = zip(*srted)  # unpack again
        clu_merged = [np.max(clu_merged)] + list(clu_merged)

        if np.all([cr.des is not None for cr in cluress.values()]):
            # all tetrodes have des
            des_merged = ["p1"]  # add mua as p1
            for cr in cluress.values():
                # ignore mua
                des_merged.extend(cr.des[1:])
        else:
            des_merged = None
        return CluRes(clu_merged, res_merged, des_merged), origins

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
        # print(phy_dir)
        cluster_groups = read_csv(join(phy_dir, "cluster_group.tsv"), sep="\t")
        spike_times = np.load(join(phy_dir, "spike_times.npy"))
        spike_clusters = np.load(join(phy_dir, "spike_clusters.npy"))
        try:
            cluster_des = read_csv(join(phy_dir, "cluster_des.tsv"), sep='\t')
        except:
            cluster_des = None
        return CluRes.__from_spikes(cluster_groups, spike_times, spike_clusters, cluster_des)

    @classmethod
    def __from_spikes(cls, cluster_groups, spike_times, spike_clusters, cluster_des=None):
        # remove spikes from unsorted clusters
        unsorted_clusters = cluster_groups[cluster_groups["group"] == "unsorted"]["cluster_id"].tolist()
        # sometimes doesn't write usnorted clusters
        # so we have to deal with that
        #all_clusters = np.unique(spike_clusters)
        #for c in all_clusters:
        #    if c not in cluster_groups.cluster_id:
        #        unsorted_clusters.append(c)

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

        des = None if cluster_des is None else ['p1']  # mua is labeled as p1
        # good clusters are from 2 onwards
        good_clusters = cluster_groups[cluster_groups["group"] == "good"]["cluster_id"].tolist()
        for i, c in enumerate(good_clusters):
            clu[spike_clusters == c] = i + 2
            if cluster_des is not None:
                des.append(cluster_des[cluster_des.cluster_id == c].des.item())
        if clu is not None and len(clu) > 0:
            clu = [clu.max()] + clu.tolist()
            assert len(clu) == len(res) + 1
        # print(good_clusters, unsorted_clusters, np.unique(clu), des)
        return CluRes(clu, res, des)
