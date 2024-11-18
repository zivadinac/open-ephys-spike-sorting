#if __name__ == '__main__' and __package__ is None:
#    from os import sys, path
#    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from os import makedirs
from os.path import join, exists, basename
from glob import glob
from argparse import ArgumentParser
import numpy as np
from spikeinterface.full import concatenate_recordings
import spikeinterface.extractors as se
from spikeinterface.core import read_binary
import spikeinterface.sorters as ss
from spikeinterface.preprocessing import bandpass_filter, common_reference
from joblib import Parallel, delayed
from src import implants, utils
from src.formats import Phy, CluRes, SUPPORTED_FORMATS
from tqdm import tqdm


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


rec2 = {}
def sort(par, recordings, out_path,
        det_thr=5, bp_min=300, bp_max=6000,
        n_jobs=1, fmt="phy", tetrodes=None):
    if not exists(out_path):
        makedirs(out_path, exist_ok=True)

    #rec = read_binary(dat_files, par["sampling_rate"], par["num_channels"],
    #                  np.uint16, is_filtered=False,
    #                  gain_to_uV=0.195, offset_to_uV=0.)
    layout_path = join(out_path, "drive_layout.txt")
    utils.save_txt([par["num_tetrodes"]] + par["tetrodes"], layout_path)
    layout = __get_implant_layout(layout_path)
    print("Concatenating recordings.")
    rec_c = concatenate_recordings(recordings).set_probegroup(layout, group_mode="by_probe")
    print(rec_c)
    rec2["rec"] = rec_c
    rec_per_tet = rec_c.split_by(property="group")
    if tetrodes is not None:
        rec_per_tet = {t: rec_per_tet[t] for t in tetrodes}

    results = __sort_tetrodes(det_thr, rec_per_tet, out_path,\
                n_jobs, fmt, bp_min, bp_max)
    __print_results(results)
    assert rec_per_tet.keys() == results.keys()


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("par_file", help="Path to the .par file.")
    args.add_argument("recording_dir", help="Path to the OpenEphys raw data.")
    args.add_argument("out_path", help="Output path for sorted data.")
    args.add_argument("--det_thr", type=float, default=5, help="Threshold for spike detection (default is 5).")
    args.add_argument("--bp_min", type=float, default=300, help="Lower threshold for bandpass filter (default is 300).")
    args.add_argument("--bp_max", type=float, default=6000, help="Upper threshold for bandpass filter (default is 6000).")
    args.add_argument("--n_jobs", type=int, default=1, help=f"Number of parallel sorting jobs to run (defauls is {1}).")
    args.add_argument("--format", default="phy", help=f"Format for the output data, one of: {SUPPORTED_FORMATS} (default is 'phy').")
    args.add_argument("--tetrodes", "-ts", type=int, default=None, nargs="+", help="List of tetrodes to process; default is None - process all tetrodes.")
    args.add_argument("--basename", default=None, help=f"Basename for the output data files.")
    args = args.parse_args()
    if args.basename is None:
        args.basename = basename(args.par_file.split('.')[0])
    utils.save_params(args)

    num_experiments = len(glob(join(args.recording_dir, "*/experiment*")))
    experiments = [se.read_openephys(args.recording_dir, block_index=e)
                   for e in range(num_experiments)]
    session_durations = []
    for ex in tqdm(experiments):
        for s in tqdm(range(ex.get_num_segments())):
            seg = ex.select_segments(s)
            session_durations.append(seg.get_total_samples())
    utils.save_txt(session_durations, join(args.out_path, f"{args.basename}.session_durations"))
    print("Saved session durations.")
    par = utils.read_par(args.par_file)
    print(f"Read {basename(args.par_file)}")
    sort(par, experiments, args.out_path,
         args.det_thr, args.bp_min, args.bp_max,
         args.n_jobs, args.format, args.tetrodes)


