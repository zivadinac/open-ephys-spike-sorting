from os import makedirs
from os.path import join, exists
from shutil import copy
from argparse import ArgumentParser
import json
import numpy as np
from pandas import read_csv
from spikeinterface.full import concatenate_recordings
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.preprocessing import bandpass_filter
from open_ephys.analysis import Session
from joblib import Parallel, delayed
import src.implants as implants
from src.formats import Phy, CluRes, SUPPORTED_FORMATS
from src.swrs import find_and_merge_SWRs
import pyopenephys as poe
import struct
from src.positions import extract_and_write_whl


def __get_implant_layout(imp):
    if exists(imp):
        return implants.read_layout(imp)
    else:
        return implants.get_implant(imp)


def __sort_tetrodes(det_thr, tet_recs, out_path, n_jobs, fmt, bp_min=300, bp_max=6000):
    def __run(tet_num, tet_recording):
        tet_fn = f"tet_{tet_num:02}"
        print(f"Sorting {tet_fn}.")
        tet_op = join(out_path, tet_fn)
        res = ss.run_sorter("mountainsort4", tet_recording,\
                      tet_op,\
                      with_output=True, remove_existing_folder=True,\
                      **{"detect_threshold": det_thr, "freq_min": bp_min, "freq_max": bp_max})
        tv_tr = tet_recording if tet_recording.get_annotation("is_filtered") else\
                bandpass_filter(tet_recording, bp_min, bp_max)
        match fmt:
            case "phy":
                Phy(tv_tr, res).save(tet_op)
            case "clu-res":
                CluRes.from_sorting(res).save(join(tet_op, tet_fn))
            case _:
                raise ValueError(f"Invalid format {fmt}. Supported formats are {SUPPORTED_FORMATS}")
        return tet_num, res

    return dict(Parallel(n_jobs=n_jobs, backend="threading")(delayed(__run)(tet_num, tet_recording) for tet_num, tet_recording in tet_recs.items()))


def __print_results(results):
    upt_str = "".join([f"\n\ttet_{i:02}: {r.get_num_units()}" for i,r in results.items()])
    print(f"Units per tet: {upt_str}")
    print(f"Total units: {np.sum([r.get_num_units() for r in results.values()])}")


if __name__ == "__main__":
    args = ArgumentParser()
    # args.add_argument("recording_path", help="Path to the recorded data.")
    # args.add_argument("drive", type=str, help=f"Path to drive layout file or name of one of predefined drives: {implants.DEFINED_IMPLANTS}.")
    # args.add_argument("out_path", help="Output path for sorted data.")
    args.add_argument("--det_thr", type=float, default=5, help="Threshold for spike detection (default is 5).")
    args.add_argument("--bp_min", type=float, default=300, help="Lower threshold for bandpass filter (default is 300).")
    args.add_argument("--bp_max", type=float, default=6000, help="Upper threshold for bandpass filter (default is 6000).")
    args.add_argument("--n_jobs", type=int, default=1, help=f"Number of parallel sorting jobs to run (defauls is {1}).")
    args.add_argument("--format", default="phy", help=f"Format for the output data, one of: {SUPPORTED_FORMATS} (default is 'phy').")
    args.add_argument("--basename", default="basename", help=f"Basename for the output data files.")
    args.add_argument("--laser_channel", type=int, default=None, help=f"TTL channel for laser pulses.")
    args.add_argument("--tracking_channel", type=int, default=None, help=f"TTL channel for laser pulses.")
    args.add_argument("--tetrodes", "-ts", type=int, default=None, nargs="+", help="List of tetrodes to process; default is None - process all tetrodes.")
    args.add_argument("--experiment_index", "-exp", type=int, default=None)
    args = args.parse_args()
    args.recording_path = "/data/jc296_tuning/jc296_020823/jc296_2023-08-02_11-15-46"
    args.drive = "igor_drive_og_rhd_64"
    args.out_path = "/data/jc296_tuning/jc296_020823_preprocess_test/"
    args.experiment_index = 0
    __write_params(args)


    print(args)
    rec = se.OpenEphysBinaryRecordingExtractor(args.recording_path,
                                               block_index=args.experiment_index)
    __extract_and_write_whl(args)
    print(rec)
    raise Exception("Done")

    resofs = __get_resofs(rec)
    desen = __read_desen(args)
    laser, laser_per_session = __read_laser(args, resofs, desen)
    try:
        swrs = find_and_merge_SWRs(args.recording_path, resofs.tolist())
    except:
        print("Cannot find SWRs (.sw files).")
        swrs = None

    __write_txt(resofs, join(args.out_path, f"{args.basename}.resofs"))
    desen.to_csv(join(args.out_path, f"{args.basename}.desen"), sep=' ', header=False, index=False)
    if laser is not None:
        np.savetxt(join(args.out_path, f"{args.basename}.laser"), laser, delimiter=' ', fmt="%i")
    if laser_per_session is not None:
        for s, ls in laser_per_session.items():
            np.savetxt(join(args.out_path, f"{args.basename}_{s}.laser"), ls, delimiter=' ', fmt="%i")
    if swrs is not None and swrs.size > 0:
        np.savetxt(join(args.out_path, f"{args.basename}.sw"), swrs, delimiter=' ', fmt="%i")

    if exists(join(args.recording_path, "recording.desel")):
        copy(join(args.recording_path, "recording.desel"),
             join(args.out_path, f"{args.basename}.desel"))

    layout = __get_implant_layout(args.drive)
    rec_c = concatenate_recordings([rec]).set_probegroup(layout, group_mode="by_probe")
    rec_per_tet = rec_c.split_by(property="group")
    if args.tetrodes is not None:
        rec_per_tet = {t: rec_per_tet[t] for t in args.tetrodes}

    results = __sort_tetrodes(args.det_thr, rec_per_tet, args.out_path,\
                args.n_jobs, args.format, args.bp_min, args.bp_max)
    __print_results(results)
    assert rec_per_tet.keys() == results.keys()
