if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from os import makedirs
from os.path import join, exists, dirname
from shutil import copy
from argparse import ArgumentParser
import json
import numpy as np
from pandas import read_csv
from spikeinterface.full import concatenate_recordings
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.preprocessing import bandpass_filter
from joblib import Parallel, delayed
import src.implants as implants
from src.formats import Phy, CluRes, SUPPORTED_FORMATS
from src.utils import read_par
import pyopenephys as poe
import struct
from pprint import pprint


def __write_params(args):
    if not exists(args.out_path):
        makedirs(args.out_path, exist_ok=True)
    with open(join(args.out_path, "params.txt"), 'w') as f:
        for arg_name, arg_val in args.__dict__.items():
            f.write(f"{arg_name}={arg_val}\n")


def __get_resofs(recording):
    shifts = [recording.get_num_samples(sn) for sn in range(recording.get_num_segments())]
    shifts = np.cumsum(shifts)
    assert shifts[-1] == recording.get_total_samples()
    return shifts


def __sort_tetrodes(det_thr, tet_recs, out_path, n_jobs, fmt, bp_min=300, bp_max=6000):
    def __run(tet_num, tet_recording):
        tet_fn = f"tet_{tet_num:02}"
        print(f"Sorting {tet_fn}.")
        tet_op = join(out_path, tet_fn)
        res = ss.run_sorter("mountainsort4", tet_recording,
                            tet_op,
                            with_output=True, remove_existing_folder=True,
                            **{"detect_threshold": det_thr,
                               "freq_min": bp_min, "freq_max": bp_max})
                               #"chunk_duration": 36000., "n_jobs": 2,
                               #"progress_bar": True})
        tv_tr = tet_recording if tet_recording.is_filtered() else\
            bandpass_filter(tet_recording, bp_min, bp_max)
        if fmt == "phy":
            Phy(tv_tr, res).save(tet_op)
        elif fmt == "clu-res":
            CluRes.from_sorting(res).save(join(tet_op, tet_fn))
        else:
            raise ValueError(f"Invalid format {fmt}. Supported formats are {SUPPORTED_FORMATS}")
        return tet_num, res
    return dict(Parallel(n_jobs=n_jobs, backend="threading")(delayed(__run)(tet_num, tet_recording) for tet_num, tet_recording in tet_recs.items()))


def __print_results(results):
    upt_str = "".join([f"\n\ttet_{i:02}: {r.get_num_units()}" for i,r in results.items()])
    print(f"Units per tet: {upt_str}")
    print(f"Total units: {np.sum([r.get_num_units() for r in results.values()])}")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("par_file", help="Path to the recorded data.")
    args.add_argument("--out_path", default=None, help="Output path for sorted data.")
    args.add_argument("--det_thr", type=float, default=5, help="Threshold for spike detection (default is 5).")
    args.add_argument("--bp_min", type=float, default=300, help="Lower threshold for bandpass filter (default is 300).")
    args.add_argument("--bp_max", type=float, default=6000, help="Upper threshold for bandpass filter (default is 6000).")
    args.add_argument("--n_jobs", type=int, default=1, help=f"Number of parallel sorting jobs to run (defauls is {1}).")
    args.add_argument("--format", default="phy", help=f"Format for the output data, one of: {SUPPORTED_FORMATS} (default is 'phy').")
    args.add_argument("--tracking_channel", type=int, default=None, help=f"TTL channel for laser pulses.")
    args.add_argument("--tetrodes", "-ts", type=int, default=None, nargs="+", help="List of tetrodes to process; default is None - process all tetrodes.")
    args = args.parse_args()
    # set output path if not provided
    if args.out_path is None:
        args.out_path = join(dirname(args.par_file), "sorting")
    __write_params(args)
    # load par file and process
    par = read_par(args.par_file)
    dat_paths = [join(par["recording_dir"], f"{s}.dat")
                 for s in par["sessions"]]
    if args.tetrodes is None:
        args.tetrodes = list(range(par["num_tetrodes"]))
    rec = se.BinaryRecordingExtractor(file_paths=dat_paths,
                                      sampling_frequency=par["sampling_rate"],
                                      num_channels=par["num_channels"],
                                      dtype=eval("np.int" + f"{par['dat_bits']}"))
    layout = implants.read_layout(par)
    rec_c = concatenate_recordings([rec])  # .set_probegroup(layout, group_mode="by_probe")
    rec_per_tet = {}
    for _ti, _tet in enumerate(layout.probes):
        if _ti in args.tetrodes:
            rec_per_tet[_ti] = rec_c.set_probe(_tet)
    #rec_per_tet = rec_c.split_by(property="group")
    #tets_to_sort = args.tetrodes if args.tetrodes is not None and len(args.tetrodes) > 0 else rec_per_tet.keys()
    #rec_per_tet = {t: rec_per_tet[t] for t in tets_to_sort}
    """
    # save individual tetrodes to disc
    makedirs(join(args.out_path, "tetrode_binary"), exist_ok=True)
    for tn, t in rec_per_tet.items():
        rec_per_tet[tn] = t.save(folder=join(args.out_path, "tetrode_binary", f"tet_{tn}"),
                                 name=f"tet_{tn}", format="binary",
                                 chunk_memory="20G", n_jobs=4, progress_bar=True)
    #rec_per_tet = {4: rec_per_tet[4], 8: rec_per_tet[8], 9: rec_per_tet[9]}
    """
    pprint(rec_per_tet)
    results = __sort_tetrodes(args.det_thr, rec_per_tet, args.out_path,
                              args.n_jobs, args.format,
                              args.bp_min, args.bp_max)
    __print_results(results)
    assert rec_per_tet.keys() == results.keys()
