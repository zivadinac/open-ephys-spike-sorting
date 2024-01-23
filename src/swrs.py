from pathlib import Path
from os.path import join
import numpy as np
from jcl.load import readfromtxt
from subprocess import run


def find_SWRs(basedir):
    """ Traverses 'basedir' saved by OpenEphys GUI to find '.sw' files.

        Args:
            basedir - dir with recorded data

        Return:
            dictionary {recording: list of SWRs}
    """
    p = Path(basedir)
    recordings = p.glob("*/experiment1/recording*")
    return {r.name: readfromtxt(list(r.glob("continuous/*/continuous.sw"))[0],
                                lambda l: [int(n) for n in l.strip().split()])
            for r in recordings}


def merge_SWRs(swrs, resofs):
    """ Merges given SWRs while shifting timestamps based on 'resofs'.

        Args:
            swrs - dictionary of SWRs {recording: list of SWRs}
            resofs - list of recording (session) ending timestamps

        Return:
            Merged SWRs, np.ndarray of shape (N, 3)
    """
    assert len(swrs) == len(resofs)
    swrs = {int(r.strip("recording")): s for r,s in swrs.items()}
    resofs = [0] + resofs
    swrs_list = []
    for i in swrs.keys():
        swrs_i = np.array(swrs[i])
        sb, se = resofs[i-1], resofs[i]
        swrs_i += sb
        assert swrs_i[0, 0] >= sb and swrs_i[-1, -1] <= se
        swrs_list.append(swrs_i)
    return np.concatenate(swrs_list, axis=0)


def find_and_merge_SWRs(basedir, resofs):
    """ Finds SWRs in 'basedir' and merges them into one list."""
    return merge_SWRs(find_SWRs(basedir), resofs)


def make_eeg_swr(dat_path, desel_path, par_path, out_path):
    sh_file = join(Path(__file__).parent, "eeg_swr.sh")
    run([sh_file, dat_path, desel_path, par_path, out_path])
