from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

args = ArgumentParser()
args.add_argument("resofs_path")
args.add_argument("--sampling_rate", default=20000, type=int)
args.add_argument("--whl_sampling_rate", default=39.0625, type=float)
args = args.parse_args()

sr = args.sampling_rate
fps = args.whl_sampling_rate
spf = int(sr // fps)  # samples per whl frame
resofs = np.loadtxt(args.resofs_path, dtype=np.uint64)

clu_path = args.resofs_path.replace("resofs", "clu")
clu = np.loadtxt(clu_path, dtype=np.uint16)
total_clusters = clu[0]
clu = clu[1:]

res_path = args.resofs_path.replace("resofs", "res")
res = np.loadtxt(res_path, dtype=np.uint64)


sess_lims = list(zip([0] + resofs[:-1].tolist(), resofs))
for sn, (sb, se) in tqdm(enumerate(sess_lims)):
    idx = np.logical_and(res > sb, res <= se)
    res_s = res[idx] - sb
    clu_s = [total_clusters] + clu[idx].tolist()
    res_s_path = res_path.replace(".res", f"_{sn+1}.res")
    clu_s_path = clu_path.replace(".clu", f"_{sn+1}.clu")
    np.savetxt(res_s_path, res_s, fmt="%d")
    np.savetxt(clu_s_path, clu_s, fmt="%d")
