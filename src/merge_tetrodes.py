from os.path import join, isdir
from os import listdir
from argparse import ArgumentParser
import numpy as np
from formats import CluRes, SUPPORTED_FORMATS


args = ArgumentParser()
args.add_argument("input_dir", help="Input directory with 'tet' subdirectories.")
args.add_argument("--format", default="phy", help=f"Format for the input data, one of: {SUPPORTED_FORMATS}.")
args.add_argument("--basename", default="basename", help=f"Basename for the merged data files.")
args = args.parse_args()

tets = listdir(args.input_dir)
tets = filter(lambda tet: tet.startswith("tet"), tets)
tets = list(filter(lambda tet: isdir(join(args.input_dir, tet)), tets))
tn_idx = 4 if '_' in tets[0] else 3
srted_tets = sorted(zip([int(tet[tn_idx:]) for tet in tets], tets)) # cut out after tet_
tet_nums, tets = zip(*srted_tets)

match args.format:
    case "phy":
        CluRess = {tet: CluRes.from_phy(join(args.input_dir, tet, "phy_export")) for tet in tets}
    case "clu-res":
        CluRess = {tet: CluRes.load(join(args.input_dir, tet, tet)) for tet in tets}
    case _:
        raise ValueError(f"Invalid --format flag, use one of: {SUPPORTED_FORMATS}.")

merged, origins = CluRes.merge_tetrodes(CluRess)
merged.save(join(args.input_dir, args.basename))

with open(join(args.input_dir, f"{args.basename}.clu_origin"), "w") as f:
    for c, tet in origins.items():
        f.write(f"{c} {tet}\n")
