#if __name__ == '__main__' and __package__ is None:
#    from os import sys, path
#    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

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
from spikeinterface.toolkit.preprocessing import bandpass_filter
from open_ephys.analysis import Session
from joblib import Parallel, delayed
import src.implants as implants
from src.formats import Phy, CluRes, SUPPORTED_FORMATS
from src.swrs import find_and_merge_SWRs
import pyopenephys as poe


def __extract_raw_positions(tracking, ttl_ts):
    x = (tracking.x * tracking.width).astype(np.int16)
    y = (tracking.y * tracking.height).astype(np.int16)
    assert x.shape == y.shape

    tracking_ts = np.array(tracking.times) - np.array(tracking.times)[0]
    ttl_ts_normed = (ttl_ts - ttl_ts[0]) * 0.05 / 1000

    whl = -1 * np.ones((len(ttl_ts), 2))
    num_missing = len(ttl_ts) - len(x)
    for i, ts in enumerate(tracking_ts):
        j = i + np.argmin(np.abs(ts - ttl_ts[i:i + num_missing]))
        whl[j] = x[i], y[i]
    return whl


def __extract_and_write_whl(args):
    sess = Session(args.recording_path)
    tr_sess = poe.File(sess.recordnodes[0].directory)

    whls = []
    for i, (r_s, r_t) in enumerate(zip(sess.recordnodes[0].recordings, tr_sess.experiments[0].recordings)):
        try:
            ttl_ts = r_s.events[(r_s.events.channel == args.tracking_channel) &
                                (r_s.events.state == 0)].timestamp.to_numpy()
            whl_r = __extract_raw_positions(r_t.tracking[0], ttl_ts)
            whl_g = __extract_raw_positions(r_t.tracking[1], ttl_ts)
            whl_b = __extract_raw_positions(r_t.tracking[2], ttl_ts)
            whl = np.concatenate([whl_r, whl_g, whl_b], axis=1)
            whls.append(whl)
            print(i, whl.shape)
        except Exception as e:
            print(e)
            print(f"Skipping recording {i}")
            if i > 0:
                raise e
    with open(join(args.out_path, f"{args.basename}.whl"), "w") as out_f:
        for whl in whls:
            np.savetxt(out_f, whl, fmt="%5d")


def __read_desen(args):
    path = join(args.recording_path, "recording.desen")

    if not exists(path):
        raise ValueError(f"Put 'recording.desen' file in the input directory {args.recording_path}")

    columns = ["session_type", "env_type", "laser_type"]
    desen = read_csv(path, names=columns, sep=' ', header=None)
    for c in columns:
        desen[c] = desen[c].str.lower()
    return desen


def __read_laser(args, resofs, desen=None):
    if args.laser_channel is None:
        return None

    if desen is None:
        desen = __read_desen(args)

    rec = Session(args.recording_path)
    laser_inds = desen[desen.laser_type != "no"].index.tolist()
    print(laser_inds)
    laser_ts = []
    for li in laser_inds:
        laser_rec = rec.recordnodes[0].recordings[li]
        laser = laser_rec.events[laser_rec.events.channel == args.laser_channel]
        laser_on = laser.timestamp[laser.state == 1].to_numpy()
        laser_off = laser.timestamp[laser.state == 0].to_numpy()
        assert laser_on[0] != laser_off[0]
        if laser_on.shape != laser_off.shape:
            if laser_on[0] < laser_off[0]:
                laser_on = laser_on[:-1]
            else:
                print("laser_off")
                laser_off = laser_off[1:]
        laser = np.stack([laser_on, laser_off], axis=1)
        # shift timestamps to start at the end of previous session
        laser = laser - laser_rec.continuous[0].timestamps[0]\
                      + (resofs[li-1] if li > 0 else 0)
        laser_ts.append(laser)
    return np.concatenate(laser_ts)


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


def __get_resofs(recording):
    shifts = [recording.get_num_samples(sn) for sn in range(recording.get_num_segments())]
    shifts = np.cumsum(shifts)
    assert shifts[-1] == recording.get_total_samples()
    return shifts


def __write_txt(ss, path):
    with open(path, 'w') as f:
        ss_str = '\n'.join(map(str, ss))
        f.write(ss_str)


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
    #args.add_argument("recording_path", help="Path to the recorded data.")
    #args.add_argument("drive", type=str, help=f"Path to drive layout file or name of one of predefined drives: {implants.DEFINED_IMPLANTS}.")
    #args.add_argument("out_path", help="Output path for sorted data.")
    args.add_argument("--det_thr", type=float, default=5, help="Threshold for spike detection (default is 5).")
    args.add_argument("--bp_min", type=float, default=300, help="Lower threshold for bandpass filter (default is 300).")
    args.add_argument("--bp_max", type=float, default=6000, help="Upper threshold for bandpass filter (default is 6000).")
    args.add_argument("--n_jobs", type=int, default=1, help=f"Number of parallel sorting jobs to run (defauls is {1}).")
    args.add_argument("--format", default="phy", help=f"Format for the output data, one of: {SUPPORTED_FORMATS} (default is 'phy').")
    args.add_argument("--basename", default="basename", help=f"Basename for the output data files.")
    args.add_argument("--laser_channel", type=int, default=None, help=f"TTL channel for laser pulses.")
    args.add_argument("--tracking_channel", type=int, default=None, help=f"TTL channel for laser pulses.")
    args = args.parse_args()
    args.recording_path = "/workspace/phd_data/raw/2022-11-23_main"
    args.drive = "igor_drive_og_rhd_64"
    args.out_path = "whl_test"
    args.tracking_channel = 1
    __write_params(args)


    rec = se.OpenEphysBinaryRecordingExtractor(args.recording_path)
    __extract_and_write_whl(args)
    exit()
    print(rec)

    resofs = __get_resofs(rec)
    desen = __read_desen(args)
    laser = __read_laser(args, resofs, desen)
    try:
        swrs = find_and_merge_SWRs(args.recording_path, resofs.tolist())
    except:
        print("Cannot find SWRs (.sw files).")
        swrs = None

    __write_txt(resofs, join(args.out_path, f"{args.basename}.resofs"))
    desen.to_csv(join(args.out_path, f"{args.basename}.desen"), sep=' ', header=False, index=False)
    if laser is not None:
        np.savetxt(join(args.out_path, f"{args.basename}.laser"), laser, delimiter=' ', fmt="%i")
    if swrs is not None and swrs.size > 0:
        np.savetxt(join(args.out_path, f"{args.basename}.sw"), swrs, delimiter=' ', fmt="%i")

    if exists(join(args.recording_path, "recording.desel")):
        copy(join(args.recording_path, "recording.desel"),
             join(args.out_path, f"{args.basename}.desel"))

    layout = __get_implant_layout(args.drive)
    rec_c = concatenate_recordings([rec]).set_probegroup(layout, group_mode="by_probe")
    rec_per_tet = rec_c.split_by(property="group")

    results = __sort_tetrodes(args.det_thr, rec_per_tet, args.out_path,\
                args.n_jobs, args.format, args.bp_min, args.bp_max)
    __print_results(results)
    assert rec_per_tet.keys() == results.keys()
