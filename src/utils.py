from os import makedirs
from os.path import join, exists, basename
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
    return np.memmap(path, dtype=np.uint16, mode='r').reshape(-1, session.nchan)
    # return np.fromfile(path, dtype=np.uint16).reshape(-1, session.nchan)


def truncate_dat_512(dat, out_path, blk_size=4194304):
    dur = dat.shape[0]
    if dur % 512 != 0:
        dur_512 = duration_512(dur)
        trc_dat = np.memmap(out_path, dtype=dat.dtype, mode="w+", shape=(dur_512, dat.shape[1]))
        for idx in np.arange(0, dur_512, blk_size):
            end = np.minimum(idx+blk_size, dur_512)
            trc_dat[idx:end, :] = dat[idx:end, :]
            trc_dat.flush()


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


def read_info(path, strict=False):
    """ Read .info file - text file with information about the recording day.
        Format of the file (all coordinates in pixels):

        pixels_per_cm (float)
        starting_box_TL_x, starting_box_TL_y, starting_box_BR_x, starting_box_BR_y
        maze_center_x, maze_center_y
        reward_new_1_x, reward_new_1_y, reward_new_2_x, reward_new_2_y, reward_new_3_x, reward_new_3_y
        reward_old_1_x, reward_old_1_y, reward_old_2_x, reward_old_2_y, reward_old_3_x, reward_old_3_y

        Args:
            path - path to the .info file
            strict - if True raise exception if cannot read the whole file,
                     if False (default) read only pixels_per_cm and
                     hide exceptions if other lines cannot be read
        Return:
            dictionary with data
    """
    with open(path, "r") as f:
        lines = f.readlines()
    info = {}
    info["pix_per_cm"] = float(lines[0].strip())
    info["sb"], info["maze_center"] = None, None
    info["rewards_new"], info["rewards_old"] = None, None
    try:
        sb = lines[1].strip().split(' ') 
        info["sb"] = [(float(sb[0]), float(sb[1])), (float(sb[2]), float(sb[3]))]
        info["maze_center"] = (float(c) for c in lines[2].strip().split(' '))
        new_r = lines[3].strip().split(' ')
        info["rewards_new"] = [(float(new_r[0]), float(new_r[1])),
                               (float(new_r[2]), float(new_r[3])),
                               (float(new_r[4]), float(new_r[5]))]
        old_r = lines[4].strip().split(' ')
        info["rewards_old"] = [(float(old_r[0]), float(old_r[1])),
                               (float(old_r[2]), float(old_r[3])),
                               (float(old_r[4]), float(old_r[5]))]
    except Exception as e:
        if strict:
            raise e
    return info


def read_par(path):
    par = {}
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [ll.strip() for ll in lines]
        par["num_channels"] = int(lines[0].strip().split(' ')[0])
        par["dat_bits"] = int(lines[0].strip().split(' ')[1])

        par["sampling_period"] = int(lines[1].strip().split(' ')[0])
        par["eeg_sampling_period"] = float(lines[1].strip().split(' ')[1])

        par["num_tetrodes"] = int(lines[2].strip().split(' ')[0])
        par["ref_channel"] = int(lines[2].strip().split(' ')[1])
        par["tetrodes"] = lines[3:3+par["num_tetrodes"]]
    sp = par["sampling_period"]
    par["sampling_rate"] = 24000 if sp == 42 else int(1_000_000 / sp)
    par["basename"] = basename(path).split('.')[0]
    return par
