#!/bin/bash
dat_path=$(readlink -f $1)
desel_path=$(readlink -f $2)
par_path=$(readlink -f $3)
out_path=$(readlink -f $4)

bn=$(basename "$1")
bn="${bn%.*}"

tmp_dir="$out_path/eeg_swr_${bn}"

mkdir -p "$tmp_dir"
cd "$tmp_dir"

ln -s "$dat_path" "input.dat"
ln -s "$desel_path" "input.desel"
ln -s "$par_path" "input.par"

#echo "Creating $bn.eeg"
sfilt3b input eeg
#echo "Detecting SWRs"
fdetswdiff input

mv input.eeg "$out_path/$bn.eeg"
mv input.sw "$out_path/$bn.sw"
cd "$out_path"
rm -rf eeg_swr
