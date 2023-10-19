from os.path import join, exists
import numpy as np
from pandas import read_csv
import json


def position_times_512(session_duration_ts):
    """ Return session-spanning timestamps that are divisible by 512.

        Args:
            session_duration_ts - original session duration (in number of samples)
        Return:
            List of timestamps
    """
    return np.arange(512, session_duration_ts + 1, 512)


def duration_512(session_duration_ts):
    """ Return session duration that is divisible by 512.

        Args:
            session_duration_ts - original session duration (in number of samples)
        Return:
            duration divisible by 512
    """
    return position_times_512(session_duration_ts)[-1]


def read_desen(path):
    columns = ["session_name", "has_laser", "laser_conf"]
    desen = read_csv(path, names=columns, sep=' ', header=None)
    for c in columns:
        desen[c] = desen[c].str.lower()
    return desen


def get_dat_path(session):
    """ Get absolute path to continuous.dat file for the given session.

        Args:
            session - poe.Recording object
        Return:
            dat file as numpy array
    """
    return join(session.absolute_foldername, "continuous/Rhythm_FPGA-100.0/continuous.dat")


def load_dat(session):
    """ Load continuous.dat file for the given session.

        Args:
            session - poe.Recording object
        Return:
            dat file as numpy array
    """
    path = get_dat_path(session)
    return np.fromfile(path, dtype=np.uint16).reshape(-1, session.nchan)


def get_resofs(session_durations_512):
    """ Get session limits for resofs file. """
    shifts = np.cumsum(session_durations_512)
    assert shifts[-1] == np.sum(session_durations_512)
    return shifts


def save_txt(ss, path):
    """ Save to a textual file. """
    with open(path, 'w') as f:
        ss_str = '\n'.join(map(str, ss)) + '\n'
        f.write(ss_str)


def save_params(args):
    if not exists(args.out_path):
        makedirs(args.out_path, exist_ok=True)

    with open(join(args.out_path, f"{args.basename}.params"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def read_info(path):
    with open(path, "r") as f:
        lines = f.readlines()
    info = {}
    info["pix_per_cm"] = float(lines[0].strip())
    info["sb"] = lines[1].strip()
    info["rewards"] = lines[2].strip()
    return info
