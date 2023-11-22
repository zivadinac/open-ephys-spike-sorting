from os.path import join
import pyopenephys as poe
import numpy as np
import struct
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import src.utils as utils
from jcl.utils import interpolate_position
from tqdm import tqdm


class MyRec:
    def __init__(self):
        self.tracking = []
        self.events = []

class MyTrk:
    def __init__(self, x, y, w, h, times):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.times = times

class MyEvs:
    def __init__(self, times, cs):
        self.times = times
        self.channel_states = cs


def __read_rec(rec_dir, sampling_rate, start_ts=0):
    rec = MyRec()

    ttl_dir = rec_dir / "events/Rhythm_FPGA-100.0/TTL_1"
    ev_cs = np.load(ttl_dir / "channel_states.npy")
    ev_ts = (np.load(ttl_dir / "timestamps.npy") - start_ts) / 20000
    rec.events.append(MyEvs(ev_ts, ev_cs))

    for i in [1, 2, 3]:
        trk_dir = rec_dir / f"events/Tracking_Port-104.0/BINARY_group_{i}"
        tr_da = np.load(trk_dir / "data_array.npy")
        tr_da = np.array([struct.unpack('4f', d) for d in tr_da])
        tr_ts = np.load(trk_dir / "timestamps.npy")
        # mitigate wrongly written timestamps by using TTL timestamps
        tr_ts = ev_ts[ev_cs == 1][:len(tr_ts)]
        trk = MyTrk(tr_da[:, 0], tr_da[:, 1], tr_da[:, 2], tr_da[:, 3], tr_ts)
        rec.tracking.append(trk)
    return rec


def __extract_raw_positions(tracking, ttl_ts, sampling_rate):
    from jcl.utils import __interpolate_linear_position
    x, y = tracking.x.copy(), tracking.y.copy()
    valid_x, valid_y = x >= 0, y >= 0

    x[valid_x] = x[valid_x] * tracking.width[valid_x]
    x[x < 0] = -1

    y[valid_y] = y[valid_y] * tracking.height[valid_y]
    y[y < 0] = -1

    x, y = x.astype(np.int16), y.astype(np.int16)
    assert x.shape == y.shape

    tracking_ts = tracking.times.magnitude if type(tracking.times) is not np.ndarray\
                  else tracking.times
    whl = -1 * np.ones((len(ttl_ts), 3), dtype=int)
    for i, ts in enumerate(tracking_ts):
        j = np.argmax(ts < ttl_ts) - 1
        whl[j] = x[i], y[i], int((ttl_ts[j]) * sampling_rate)
    whl[:, 2] = __interpolate_linear_position(whl[:, 2]).astype(int)
    return whl


def extract_raw_whl(sessions, sampling_rate=20000):
    whls = []
    print("Extracting whl data.")
    for i, r_t in tqdm(enumerate(sessions)):
        if r_t._processor_sample_rates is None or len(r_t._processor_sample_rates) == 0:
            print(f"__read_rec {i+1}")
            r_t_start_ts = int(r_t.start_time.magnitude * sampling_rate)
            r_t = __read_rec(r_t.absolute_foldername, sampling_rate, r_t_start_ts)

        ttl_ts = r_t.events[0].times[r_t.events[0].channel_states == 1]
        if type(ttl_ts) is not np.ndarray:
            ttl_ts = ttl_ts.magnitude

        whl_r = __extract_raw_positions(r_t.tracking[0], ttl_ts, sampling_rate)
        whl_g = __extract_raw_positions(r_t.tracking[1], ttl_ts, sampling_rate)
        whl_b = __extract_raw_positions(r_t.tracking[2], ttl_ts, sampling_rate)
        whl = np.concatenate([whl_r[:, [0,1]], whl_g[:, [0,1]], whl_b], axis=1)
        whls.append(whl)
    return whls


def resample_512(whl, duration_ts, pix_per_cm=1, interp_order=1):
    """ Spline interpolation of missing values in animal position data.

        Args:
            whl - position data, the last column should contain timestamps
            duration_ts - session duration (total numbe of samples)
            pix_per_cm - needed to conver to cm (default is 1, i.e. no conversion)
            interp_order - interpolation polynom degree, default 1 (linear)
        Return:
            Interpolated data in the same format as provided.
    """
    orig_ts = whl[:, -1]
    new_ts = utils.position_times_512(duration_ts)

    new_whl = -1 * np.ones((len(new_ts), whl.shape[1]))
    new_whl[:, -1] = new_ts
    for c in range(whl.shape[1]-1):
        position = np.array(whl[:, c])
        position[position <= 0] = -1
        ok_idx = np.where(position != -1)[0]

        if position[0] == -1:
            position[0] = int(np.mean(position[ok_idx[:10]]))
        if position[-1] == -1:
            position[-1] = int(np.mean(position[ok_idx[-10:]]))

        position = interpolate_position(position)
        f = UnivariateSpline(orig_ts, position, k=interp_order)
        # prevent interpolation from going crazy
        p_l, p_h = position[ok_idx].min(), position[ok_idx].max()
        res_position = np.clip(f(new_ts), p_l, p_h)

        # convert to cm
        if pix_per_cm != 1:
            res_position = res_position / pix_per_cm

        new_whl[:, c] = res_position
    return new_whl


def merge_whls(whls, session_durations):
    shifted_whls = []
    shifts = np.cumsum([0] + session_durations[:-1])
    for whl, ss in zip(whls, shifts):
        s_whl = whl.copy()
        s_whl[:, -1] = s_whl[:, -1] + ss
        shifted_whls.append(s_whl)
    return np.vstack(shifted_whls)


def extract_whl(sessions, session_durations, pix_per_cm):
    """ Extract position data into whl form.

        Args:
            sessions - list of poe.Recording objects
            session_durations - duration of each session (number of samples)
            pix_per_cm - needed to convert to cm (default is 1, i.e. no conversion)
        Return:
            ([raw_whls], raw_whls_merged), ([resampled_whls], resampled_whls_merged)
    """
    whls = extract_raw_whl(sessions)
    whls_m = merge_whls(whls, session_durations)

    res_whls = [resample_512(whl, sd, pix_per_cm) for whl, sd in zip(whls, session_durations)]
    res_whls_m = merge_whls(res_whls, [res_whl[-1, -1] for res_whl in res_whls])
    return (whls, whls_m), (res_whls, res_whls_m)


def __save_whl(out_path, basename, whls, whl_merged=None, ext="whl", fmt="%.3f"):
    for i, whl in enumerate(whls):
        with open(join(out_path, f"{basename}_{i + 1}.{ext}"), "w") as out_f:
            np.savetxt(out_f, whl, fmt=fmt)
    if whl_merged is not None:
        with open(join(out_path, f"{basename}.{ext}"), "w") as out_f:
            np.savetxt(out_f, whl_merged, fmt=fmt)


def save_whls(out_path, basename, raw_whls, res_whls):
    """ Save whls to disc.

        Args:
            out_path - path to output directory
            basename - basename to use for files
            raw_whls - ([raw_whl per session], merged_raw_whl)
            res_whls - ([resampled_whl per session], merged_resampled_whl)
    """
    __save_whl(out_path, basename, *raw_whls, ext="whl.raw", fmt="%5d")
    __save_whl(out_path, basename, *res_whls)


