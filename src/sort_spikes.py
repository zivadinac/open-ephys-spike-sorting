if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from os import makedirs
from os.path import join, exists
from argparse import ArgumentParser
import json
import numpy as np
from spikeinterface.full import concatenate_recordings
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.core import load_extractor
from joblib import Parallel, delayed
import src.implants as implants


def __write_params(args):
    if not exists(args.out_path):
        makedirs(args.out_path, exist_ok=True)

    with open(join(args.out_path, "params.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def __get_implant_layout(imp):
    if exists(imp):
        return implants.read_layout(imp)
    else:
        return implants.get_implant(imp)


def __get_segment_shifts(recording):
    shifts = [recording.get_num_samples(sn) for sn in range(recording.get_num_segments())]
    shifts = np.cumsum(shifts)
    assert shifts[-1] == recording.get_total_samples()
    return shifts[:-1]


def __write_segment_shifts(ss, path):
    with open(path, 'w') as f:
        ss_str = '\n'.join(map(str, ss))
        f.write(ss_str)


def __sort_tetrodes(det_thr, tet_recs, out_path, n_jobs):
    def __run(tn, tr, dt, op):
        print(f"Sorting tetrode {tn}.")
        return tn, ss.run_sorter("mountainsort4", load_extractor(tr),\
                      join(op, f"tet{tn}"),\
                      with_output=True, remove_existing_folder=True,\
                      **{"detect_threshold": dt})

    return dict(Parallel(n_jobs=n_jobs, backend="threading")(delayed(__run)(tn, tr.to_dict(), det_thr, out_path) for tn, tr in tet_recs.items()))


def __print_results(results):
    upt_str = "".join([f"\n\ttet{i}: {r.get_num_units()}" for i,r in results.items()])
    print(f"Units per tet: {upt_str}")
    print(f"Total units: {np.sum([r.get_num_units() for r in results.values()])}")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("recording_path", help="Path to the recorded data.")
    args.add_argument("drive", type=str, help=f"Path to drive layout file or name of one of predefined drives: {implants.DEFINED_IMPLANTS}.")
    args.add_argument("out_path", help="Output path for sorted data.")
    args.add_argument("--det_thr", type=float, default=5, help="Threshold for spike detection (default is 5).")
    args.add_argument("--n_jobs", type=int, default=1, help=f"Number of parallel sorting jobs to run (defauls is {1}).")
    args = args.parse_args()
    __write_params(args)


    rec = se.OpenEphysBinaryRecordingExtractor(args.recording_path)
    __write_segment_shifts(__get_segment_shifts(rec), join(args.out_path, "merged.resofs"))

    layout = __get_implant_layout(args.drive)
    rec_c = concatenate_recordings([rec]).set_probegroup(layout, group_mode="by_probe")
    rec_per_tet = rec_c.split_by(property="group")

    results = __sort_tetrodes(args.det_thr, rec_per_tet, args.out_path, args.n_jobs)
    __print_results(results)
