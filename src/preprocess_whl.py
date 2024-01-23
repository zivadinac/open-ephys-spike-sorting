from glob import glob
from os.path import join, basename
from argparse import ArgumentParser
import numpy as np
import jcl
from scipy.signal import savgol_filter


def __get_sess_name(whl_name):
    sn = whl_name.split('.')[0]
    return int(sn.split('_')[-1])


def __preprocess_whl(whl, size=None, pixels_per_cm=1., filter_window=20):
    """ Preprocess raw whl.
        
        Args:

            whl - whl to preprocess, two columns X and Y, shape (N, 2)
            size - frame size in pixels (width, height), used for flipping around y-axis;
                   if None (default) it will be set to capture raw whl
            pixels_per_cm - used for converting trajectory units from pixels to cm;
                            default is 1 (no conversion)
            filter_window - window length for Savitzky-Golay filter
        Return:
            Processed whl, array of the same shape as `whl`
    """
    whl[whl <= 0] = -1

    known_idx = np.where(np.any(whl != [-1, -1], axis=1))[0]
    if known_idx[0] > 0:
        whl[0] = whl[known_idx[0]]

    if known_idx[-1] < len(whl) - 1:
        whl[-1] = whl[known_idx[-1]]

    x_l = whl[:, 0][whl[:, 0] > 0].min()
    x_h = whl[:, 0][whl[:, 0] > 0].max()
    y_l = whl[:, 1][whl[:, 1] > 0].min()
    y_h = whl[:, 1][whl[:, 1] > 0].max()

    pos = jcl.utils.interpolate_position(whl[:, 0], whl[:, 1])
    pos = [np.clip(pos[0], x_l, x_h), np.clip(pos[1], y_l, y_h)]
    if size is None:
        size = (int(np.max(pos[0]) + 1), int(np.max(pos[1]) + 1))
    pos = np.stack([pos[0], size[1] - pos[1]], axis=1)
    pos = pos / pixels_per_cm
    return np.stack([savgol_filter(pos[:, 0], filter_window, 2),
                     savgol_filter(pos[:, 1], filter_window, 2)], axis=1)


args = ArgumentParser()
args.add_argument("dir")
args.add_argument("basename")
args.add_argument("--frame_size", "-ms", nargs=2, type=int, default=[1440, 1080])
args.add_argument("--pixels_per_cm", "-ppcm", type=float, default=7.8)
args = args.parse_args()

files = glob(join(args.dir, f"{args.basename}_*.whl.raw"))

for f in files:
    try:
        raw_whl = jcl.load.positions_from_whl(f)
        whl_r = __preprocess_whl(raw_whl[:, [0, 1]], args.frame_size, args.pixels_per_cm)
        whl_g = __preprocess_whl(raw_whl[:, [2, 3]], args.frame_size, args.pixels_per_cm)
        whl_b = __preprocess_whl(raw_whl[:, [4, 5]], args.frame_size, args.pixels_per_cm)
        whl_ts = raw_whl[:, -1].astype(int)
        assert len(whl_r) == len(whl_g) == len(whl_b) == len(whl_ts)
        new_path = f.replace(".raw", '')
        #orig_num = int(basename(f).split('.')[0].split('_')[-1])
        #new_path = join(args.dir, f"{args.basename}_{orig_num+1}.whl")
        with open(new_path, "w") as of:
            for i in range(len(whl_r)):
                of.write(f"{whl_r[i, 0]} {whl_r[i, 1]} {whl_g[i, 0]} {whl_g[i, 1]} {whl_b[i, 0]} {whl_b[i, 1]} {whl_ts[i]:0d}\n")
    except Exception as e:
        print(f"error with {f}")
        print(e)
        print(f"error with {f}")
        continue

#all_whls = np.concatenate(all_whls, axis=0)
#np.savetxt(join(args.dir, args.basename + ".whl"), all_whls, fmt="%8d")
