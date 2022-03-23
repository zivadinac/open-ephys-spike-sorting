from os.path import join
from spikeinterface.exporters import export_to_phy
from spikeinterface.core.waveform_extractor import WaveformExtractor
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt


def convert_to_phy(recording, sorting, out_dir):
    """ Convert sorted data to phy format.
        
        Args:
            recording - recording extractor (has to be filtered)
            sorting - sorting extractor
            out_dir - output dir (folders `wfe` and `phy_export` will appear here)
    """
    wfe = WaveformExtractor.create(recording, sorting, join(out_dir, "wfe"))
    wfe.set_params()
    wfe.run_extract_waveforms()
    sw.plot_unit_waveforms(wfe)
    plt.savefig(join(out_dir, "wfe", "waveforms.png"))
    export_to_phy(wfe, join(out_dir, "phy_export"))#, copy_binary=False)

