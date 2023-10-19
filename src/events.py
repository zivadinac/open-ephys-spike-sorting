import numpy as np
from os.path import join
from open_ephys.analysis import Session
import pyopenephys as poe


def read_laser(desen, resofs, laser_channel, sessions, session_durations_512):
    if laser_channel is None:
        return None, None

    laser_inds = desen[desen.has_laser != "no"].index.tolist()
    laser_ts = []
    laser_ts_per_session = {}
    for li in laser_inds:
        laser_rec = sessions[li]
        sr = int(laser_rec.sample_rate)
        laser_evs = [ev for ev in laser_rec.events if laser_channel in ev.channels]
        assert len(laser_evs) == 1
        laser_evs = laser_evs[0]
        laser_on = (laser_evs.times[laser_evs.channel_states == 1].magnitude * sr).astype(int)
        laser_off = (laser_evs.times[laser_evs.channel_states == -1].magnitude * sr).astype(int)
        assert laser_on[0] != laser_off[0]
        if laser_on.shape != laser_off.shape:
            if laser_on[0] < laser_off[0]:
                laser_on = laser_on[:-1]
            else:
                print("laser_off")
                laser_off = laser_off[1:]

        # only ones that had finished before the session ended
        idx = laser_off < session_durations_512[li]
        laser_on, laser_off = laser_on[idx], laser_off[idx]
        assert np.all((laser_off - laser_on) > 0)

        laser = np.stack([laser_on, laser_off], axis=1)
        laser_ts_per_session[li + 1] = laser  # session nums are 1-based
        laser = laser + (resofs[li-1] if li > 0 else 0)
        laser_ts.append(laser)
    return np.concatenate(laser_ts), laser_ts_per_session


def save_events(out_path, basename, ext, event_ts, event_ts_per_session):
    if event_ts is not None:
        np.savetxt(join(out_path, f"{basename}.{ext}"), event_ts, delimiter=' ', fmt="%i")
    if event_ts_per_session is not None:
        for s, ls in event_ts_per_session.items():
            np.savetxt(join(out_path, f"{basename}_{s}.{ext}"), ls, delimiter=' ', fmt="%i")


