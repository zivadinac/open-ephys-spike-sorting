if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from os import makedirs, link, symlink
from os.path import join, exists, basename
from shutil import copy
from argparse import ArgumentParser
import numpy as np
from src import positions, events, utils, swrs
from src.sort_preprocessed import sort
import pyopenephys as poe
from tqdm import tqdm


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("recording_path", help="Path to the recorded data.")
    args.add_argument("out_path", help="Output path for sorted data.")
    args.add_argument("--basename", default=None, help=f"Basename for the output data files.")
    args.add_argument("--laser_channel", type=int, default=None, help=f"TTL channel for laser pulses.")
    args.add_argument("--copy_all_dats", type=int, default=0, help="If 0 (default), *.dat files divisible by 512 will be linked to, others will be truncated and copied. If 1, all *.dat files will be copied.")
    args.add_argument("--skip_dats", type=int, default=0, help="If 1 no *.dat files will appear in output (default is 0).")
    args.add_argument("--sort", type=int, default=0, help="Start sorting data (default is False).")
    args = args.parse_args()
    #args.recording_path = "/data/jc296_tuning/jc296_020823/jc296_020823"
    #args.out_path = "/data/jc296_tuning/jc296_020823_preprocess_test/"
    #args.recording_path = "/data/jc296_tuning/jc296_250723"
    #args.out_path = "/data/jc296_sorting/"
    #args.laser_channel = 4
    #args.sort = False
    if args.basename is None:
        args.basename = basename(args.recording_path)
    print(args.basename)
    utils.save_params(args)


    # copy necessary configuration files
    for ext in ["par", "desen", "info", "desel"]:
        path = join(args.recording_path, f"{args.basename}.{ext}")
        assert_msg = f"Put {args.basename}.{ext} in the input directory."
        assert exists(path), assert_msg
        copy(path, join(args.out_path, f"{args.basename}.{ext}"))
        print(f"Copied {args.basename}.{ext}")

    desen = utils.read_desen(join(args.recording_path, f"{args.basename}.desen"))

    # load all recording sessions
    # they must match desen
    rec_day = poe.File(join(args.recording_path, "Record Node 101"))
    sessions = []
    for ex_i, ex in enumerate(rec_day.experiments):
        sessions.extend(ex.recordings)
    assert len(sessions) == len(desen), "Number of recording sessions must match number of lines in {args.basename}.desen"
    session_durations, session_durations_512 = [], []
    out_dat_paths = []
    print("Truncating dat files to be divisible by 512.")
    for si, s in tqdm(enumerate(sessions)):
        dat = utils.load_dat(s)

        dur = dat.shape[0]
        session_durations.append(dur)

        dur_512 = utils.duration_512(dur)
        session_durations_512.append(dur_512)

        out_dat_path = join(args.out_path, f"{args.basename}_{si+1}.dat")
        out_dat_paths.append(out_dat_path)
        if not args.skip_dats:
            if args.copy_all_dats:
                if dur == dur_512:
                    copy(utils.get_dat_path(s), out_dat_path)
                else:
                    utils.truncate_dat_512(dat, out_dat_path)
            else:
                if dur == dur_512:
                    symlink(utils.get_dat_path(s), out_dat_path)
                else:
                    utils.truncate_dat_512(dat, out_dat_path)

    # compute and save .resofs
    resofs = utils.get_resofs(session_durations_512)
    utils.save_txt(resofs, join(args.out_path, f"{args.basename}.resofs"))
    print(f"Generated {args.basename}.resofs")

    # extract and save laser timestamps
    if args.laser_channel is not None:
        laser, laser_per_session = events.read_laser(desen, resofs, args.laser_channel,
                                                     sessions, session_durations_512)
        events.save_events(args.out_path, args.basename, "light",
                           laser, laser_per_session)
        print(f"Extracted laser timestamps ({args.basename}.light)")

    # extract and save .whl
    info = utils.read_info(join(args.recording_path, f"{args.basename}.info"))
    raw_whls, res_whls = positions.extract_whl(sessions, session_durations, info["pix_per_cm"])
    positions.save_whls(args.out_path, args.basename, raw_whls, res_whls)
    print(f"Extracted positions ({args.basename}.whl)")

    """
    par = utils.read_par(join(args.recording_path, f"{args.basename}.par"))
    print("Generating .eeg and .sw")
    for dp in tqdm(out_dat_paths):
        desel_path = join(args.out_path, f"{args.basename}.desel")
        par_path = join(args.out_path, f"{args.basename}.par")
        swrs.make_eeg_swr(dp, desel_path, par_path, args.out_path)
    """

    if args.sort:
        sort(par, out_dat_paths, join(args.out_path, "sorting"))
