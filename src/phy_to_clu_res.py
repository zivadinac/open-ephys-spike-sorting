from os import listdir
from os.path import join, isdir
from argparse import ArgumentParser
from formats import CluRes

args = ArgumentParser()
args.add_argument("input_dir", help="Input directory with phy-processed data.")
args = args.parse_args()

for tet in listdir(args.input_dir):
    tet_dir = join(args.input_dir, tet)
    if not isdir(tet_dir):
        continue
    phy_dir = join(tet_dir, "phy_export")
    CluRes.from_phy(phy_dir).save(join(tet_dir, tet))
    print(f"Converted {tet} to clu-res.")
