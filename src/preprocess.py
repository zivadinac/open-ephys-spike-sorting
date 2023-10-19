if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from os import makedirs
from os.path import join, exists, basename
from shutil import copy
from argparse import ArgumentParser
import numpy as np
from pandas import read_csv
from spikeinterface.full import concatenate_recordings
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.preprocessing import bandpass_filter
from joblib import Parallel, delayed
import src.implants as implants
from src.formats import Phy, CluRes, SUPPORTED_FORMATS
from src.swrs import find_and_merge_SWRs
from src import positions, events, utils
import pyopenephys as poe


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("recording_path", help="Path to the recorded data.")
    args.add_argument("out_path", help="Output path for sorted data.")
    args.add_argument("--basename", default=None, help=f"Basename for the output data files.")
    args.add_argument("--laser_channel", type=int, default=None, help=f"TTL channel for laser pulses.")
    args = args.parse_args()
    # args.recording_path = "/data/jc296_tuning/jc296_020823/jc296_020823"
    # args.out_path = "/data/jc296_tuning/jc296_020823_preprocess_test/"
    # args.laser_channel = 4
    if args.basename is None:
        args.basename = basename(args.recording_path)
    utils.save_params(args)


    # load and copy .info
    info = utils.read_info(join(args.recording_path, f"{args.basename}.info"))
    copy(join(args.recording_path, f"{args.basename}.info"),
         join(args.out_path, f"{args.basename}.info"))

    # load and copy .desen
    desen = utils.read_desen(join(args.recording_path, f"{args.basename}.desen"))
    copy(join(args.recording_path, f"{args.basename}.desen"),
         join(args.out_path, f"{args.basename}.desen"))

    # load all recording sessions
    # they must match desen
    rec_day = poe.File(join(args.recording_path, "Record Node 101"))
    sessions = []
    for ex_i, ex in enumerate(rec_day.experiments):
        sessions.extend(ex.recordings)
    assert len(sessions) == len(desen)
    session_durations = [utils.load_dat(s).shape[0] for s in sessions]
    # round session durations to be divisible by 512
    session_durations_512 = [utils.duration_512(sd) for sd in session_durations]

    # compute and save .resofs
    resofs = utils.get_resofs(session_durations_512)
    utils.save_txt(resofs, join(args.out_path, f"{args.basename}.resofs"))

    # extract and save laser timestamps
    laser, laser_per_session = events.read_laser(desen, resofs, args.laser_channel,
                                                 sessions, session_durations_512)
    events.save_events(args.out_path, args.basename, "light",
                       laser, laser_per_session)

    # extract and save .whl
    raw_whls, res_whls = positions.extract_whl(sessions, session_durations, info["pix_per_cm"])
    positions.save_whls(args.out_path, args.basename, raw_whls, res_whls)

    # copy .desel if exists
    if exists(join(args.recording_path, f"{args.basename}.desel")):
        copy(join(args.recording_path, f"{args.basename}.desel"),
             join(args.out_path, f"{args.basename}.desel"))
