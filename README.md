# open-ephys-spike-sorting
Spike sorting pipeline for data recorded using OpenEphys hardware and GUI.

## Installation instructions

TBD

## Usage

* Preprocess data to JC lab format:
  ```python src/preprocess.py -h``` to display help.
  This script will:
    * Ask you to make .par, .desel, .desen and .info files
    * Cut .dat files to be divisible by 512
    * Extract .whl (in cm @ 39.0625Hz)
    * Save .whl.raw files (raw positions)

* Run sorting on preprocessed data
  ```python src/sort_from_par.py -h``` to display help.
