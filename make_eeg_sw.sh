main_dir=$(readlink -f $1)
cd $main_dir

# expand paths to all recordings
full_paths=$(ls */experiment*/recording*/conti*/* | grep recording)
# use : as a delimiter between different paths, because they contain whitespaces
full_paths=$(echo $full_paths | sed "s/: /:/g")
# put paths into an array (using ':' as delimiter)
IFS=":" read -ra cont_dirs <<< "$full_paths"

for d in "${cont_dirs[@]}";
do
    echo "$d"
    mkdir -p "$d/eeg_swr"
    cd "$d/eeg_swr"

    # create links to input files
    ln -s "$main_dir/recording.desel" "continuous.desel"
    ln -s "$main_dir/recording.par" "continuous.par"
    ln -s "../continuous.dat" "continuous.dat"

    echo "Creating .eeg"
    sfilt3b continuous eeg
    echo "Detecting SWRs"
    fdetswdiff continuous

    # move results to the recording dir and remove junk
    mv continuous.eeg "../"
    mv continuous.sw "../"
    cd $main_dir
    rm -rf "$d/eeg_swr"
done
