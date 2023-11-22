import numpy as np
from probeinterface import ProbeGroup, generate_tetrode
from probeinterface.io import read_prb, write_prb, write_probeinterface, write_probeinterface
import matplotlib.pyplot as plt
from probeinterface.plotting import plot_probe_group


def make_tetrode(num, channels, position=None, contacts=[[0,0],[-20,20],[20,20],[0,40]], radius=12):
    """ Make a tetrode with given params.

        Args:
            num - number (will be used as the name)
            channels - list of channels that correspond to this tetrode
            position - position of tetrode inside a drive (in microns, [AP, ML]); by default None, if given all contacts will be translated by this amount
            contacts - 2d list of electrode tips positions (in microns)
            radius - radius of electrode tip (in microns)

        Return:
            tetrode object from probeinterface
    """
    tet = generate_tetrode()
    tet.set_device_channel_indices(channels)
    tet.set_contacts(contacts, shape_params={"radius": radius})

    if position is not None:
        # the first coordinate seems to be mid-lateral axis
        # while the second is anterior-posterior axis
        # I write coords the other way around
        # so I have to reverse them here
        tet.move(list(reversed(position)))

    tet.annotate(name=str(num))
    return tet


def small_drive_rhd_64():
    """ Generate layout for JC group small drive (16 tetrodes).
        Channel layout is for Intan RHD 64-channel headstage."""
    tetrodes = []
    tetrodes.append(make_tetrode(0, [48,49,50,51], position=[-4650, 2900]))
    tetrodes.append(make_tetrode(1, [52,53,54,55], position=[-4300, 2700]))
    tetrodes.append(make_tetrode(2, [56,57,58,59], position=[-4650, 2500]))
    tetrodes.append(make_tetrode(3, [60,61,62,63], position=[-4300, 2300]))

    tetrodes.append(make_tetrode(4, [0,1,2,3], position=[-3950, 2100]))
    tetrodes.append(make_tetrode(5, [4,5,6,7], position=[-3600, 1900]))
    tetrodes.append(make_tetrode(6, [8,9,10,11], position=[-3950, 2500]))
    tetrodes.append(make_tetrode(7, [12,13,14,15], position=[-3600, 2300]))

    tetrodes.append(make_tetrode(8, [16,17,18,19], position=[-3600, 2700]))
    tetrodes.append(make_tetrode(9, [20,21,22,23], position=[-3950, 2900]))
    tetrodes.append(make_tetrode(10, [24,25,26,27], position=[-3600, 3100]))
    tetrodes.append(make_tetrode(11, [28,29,30,31], position=[-3950, 3300]))

    tetrodes.append(make_tetrode(12, [32,33,34,35], position=[-4300, 3500]))
    tetrodes.append(make_tetrode(13, [36,37,38,39], position=[-4650, 3700]))
    tetrodes.append(make_tetrode(14, [40,41,42,43], position=[-4300, 3100]))
    tetrodes.append(make_tetrode(15, [44,45,46,47], position=[-4650, 3300]))

    layout = ProbeGroup()
    for tet in tetrodes:
        layout.add_probe(tet)
    return layout


def small_drive_og_rhd_64():
    """ Generate layout for JC group small drive for optogenetics (14 tetrodes and 2 optic fibers).
        Channel layout is for Intan RHD 64-channel headstage.
        Optic fibers are in place of top right and bottom left long shuttles, e.g.
        in place of the third tetrode in the first row and the second tetrode in the last row."""
    tetrodes = []
    tetrodes.append(make_tetrode(0, [52,53,54,55], position=[-4300, 2700]))
    tetrodes.append(make_tetrode(1, [56,57,58,59], position=[-4650, 2500]))
    tetrodes.append(make_tetrode(2, [60,61,62,63], position=[-4300, 2300]))

    tetrodes.append(make_tetrode(3, [0,1,2,3], position=[-3950, 2100]))
    tetrodes.append(make_tetrode(4, [4,5,6,7], position=[-3600, 1900]))
    tetrodes.append(make_tetrode(5, [8,9,10,11], position=[-3950, 2500]))
    tetrodes.append(make_tetrode(6, [12,13,14,15], position=[-3600, 2300]))

    tetrodes.append(make_tetrode(7, [20,21,22,23], position=[-3950, 2900]))
    tetrodes.append(make_tetrode(8, [24,25,26,27], position=[-3600, 3100]))
    tetrodes.append(make_tetrode(9, [28,29,30,31], position=[-3950, 3300]))

    tetrodes.append(make_tetrode(10, [32,33,34,35], position=[-4300, 3500]))
    tetrodes.append(make_tetrode(11, [36,37,38,39], position=[-4650, 3700]))
    tetrodes.append(make_tetrode(12, [40,41,42,43], position=[-4300, 3100]))
    tetrodes.append(make_tetrode(13, [44,45,46,47], position=[-4650, 3300]))

    layout = ProbeGroup()
    for tet in tetrodes:
        layout.add_probe(tet)
    return layout


def big_drive_og_rhd_64():
    """ Generate layout for JC group's big drive for optogenetics (14x2 tetrodes and 2x2 optic fibers, bilaterally).
        Channel layout is for Intan RHD 64-channel headstage.
        Optic fibers are in place of top right and bottom left long shuttles, e.g.
        in place of the third tetrode in the first row and the second tetrode in the last row."""
    tetrodes = []
    tetrodes.append(make_tetrode(0, [12,13,14,15], position=[-3600, 1900]))
    tetrodes.append(make_tetrode(1, [8,9,10,11], position=[-3600, 2300]))
    tetrodes.append(make_tetrode(2, [20,21,22,23], position=[-3600, 3100]))

    tetrodes.append(make_tetrode(3, [4,5,6,7], position=[-3950, 2100]))
    tetrodes.append(make_tetrode(4, [0,1,2,3], position=[-3950, 2500]))
    tetrodes.append(make_tetrode(5, [24,25,26,27], position=[-3950, 2900]))
    tetrodes.append(make_tetrode(6, [28,29,30,31], position=[-3950, 3300]))

    tetrodes.append(make_tetrode(7, [60,61,62,63], position=[-4300, 2300]))
    tetrodes.append(make_tetrode(8, [56,57,58,59], position=[-4300, 2700]))
    tetrodes.append(make_tetrode(9, [32,33,34,35], position=[-4300, 3100]))
    tetrodes.append(make_tetrode(10, [36,37,38,39], position=[-4300, 3500]))

    tetrodes.append(make_tetrode(11, [52,53,54,55], position=[-4650, 2500]))
    tetrodes.append(make_tetrode(12, [40,41,42,43], position=[-4650, 3300]))
    tetrodes.append(make_tetrode(13, [44,45,46,47], position=[-4650, 3700]))

    tetrodes.append(make_tetrode(14, [80,81,82,83], position=[-3600, -1900]))
    tetrodes.append(make_tetrode(15, [84,85,86,87], position=[-3600, -2300]))
    tetrodes.append(make_tetrode(16, [72,73,74,75], position=[-3600, -3100]))

    tetrodes.append(make_tetrode(17, [88,89,90,91], position=[-3950, -2100]))
    tetrodes.append(make_tetrode(18, [92,93,94,95], position=[-3950, -2500]))
    tetrodes.append(make_tetrode(19, [68,69,70,71], position=[-3950, -2900]))
    tetrodes.append(make_tetrode(20, [64,65,66,67], position=[-3950, -3300]))

    tetrodes.append(make_tetrode(21, [96,97,98,99], position=[-4300, -2300]))
    tetrodes.append(make_tetrode(22, [100,101,102,103], position=[-4300, -2700]))
    tetrodes.append(make_tetrode(23, [124,125,126,127], position=[-4300, -3100]))
    tetrodes.append(make_tetrode(24, [120,121,122,123], position=[-4300, -3500]))

    tetrodes.append(make_tetrode(25, [104,105,106,107], position=[-4650, -2500]))
    tetrodes.append(make_tetrode(26, [116,117,118,119], position=[-4650, -3300]))
    tetrodes.append(make_tetrode(27, [112,113,114,115], position=[-4650, -3700]))

    layout = ProbeGroup()
    for tet in tetrodes:
        layout.add_probe(tet)
    return layout


def igor_drive_og_rhd_64():
    """ Generate layout for JC group's big drive for optogenetics (14x2 tetrodes and 2x2 optic fibers, bilaterally).
        Channel layout is for Intan RHD 64-channel headstage.
        Optic fibers are in place of top right and bottom left long shuttles, e.g.
        in place of the third tetrode in the first row and the second tetrode in the last row."""
    tetrodes = []
    tetrodes.append(make_tetrode(0, [12,13,14,15], position=[-3600, 1900]))
    tetrodes.append(make_tetrode(1, [16,17,18,19], position=[-3950, 2100]))
    tetrodes.append(make_tetrode(2, [8,9,10,11], position=[-3600, 2300]))
    tetrodes.append(make_tetrode(3, [20,21,22,23], position=[-3950, 2500]))

    tetrodes.append(make_tetrode(4, [4,5,6,7], position=[-3600, 3100]))
    tetrodes.append(make_tetrode(5, [24,25,26,27], position=[-3950, 2900]))
    tetrodes.append(make_tetrode(6, [0,1,2,3], position=[-3600, 2700]))
    tetrodes.append(make_tetrode(7, [28,29,30,31], position=[-3950, 3300]))

    tetrodes.append(make_tetrode(8, [60,61,62,63], position=[-4300, 3500]))
    tetrodes.append(make_tetrode(9, [32,33,34,35], position=[-4650, 3100]))
    tetrodes.append(make_tetrode(10, [56,57,58,59], position=[-4650, 3500]))
    tetrodes.append(make_tetrode(11, [36,37,38,39], position=[-4650, 3900]))

    tetrodes.append(make_tetrode(12, [52,53,54,55], position=[-4300, 3100]))
    tetrodes.append(make_tetrode(13, [40,41,42,43], position=[-4650, 3500]))
    tetrodes.append(make_tetrode(14, [48,49,50,51], position=[-4300, 2700]))
    tetrodes.append(make_tetrode(15, [44,45,46,47], position=[-4300, 2300]))

    tetrodes.append(make_tetrode(16, [112,113,114,115], position=[-3600, -1900]))
    tetrodes.append(make_tetrode(17, [108,109,110,111], position=[-3600, -2300]))
    tetrodes.append(make_tetrode(18, [116,117,118,119], position=[-3600, -3100]))
    tetrodes.append(make_tetrode(19, [104,105,106,107], position=[-3600, -3100]))

    tetrodes.append(make_tetrode(20, [120,121,122,123], position=[-3950, -2100]))
    tetrodes.append(make_tetrode(21, [100,101,102,103], position=[-3950, -2500]))
    tetrodes.append(make_tetrode(22, [124,125,126,127], position=[-3950, -2900]))
    tetrodes.append(make_tetrode(23, [96,97,98,99], position=[-3950, -3300]))

    tetrodes.append(make_tetrode(24, [64,65,66,67], position=[-4300, -2300]))
    tetrodes.append(make_tetrode(25, [92,93,94,95], position=[-4300, -2700]))
    tetrodes.append(make_tetrode(26, [68,69,70,71], position=[-4300, -3100]))
    tetrodes.append(make_tetrode(27, [88,89,90,91], position=[-4300, -3500]))

    tetrodes.append(make_tetrode(28, [72,73,74,75], position=[-4650, -2500]))
    tetrodes.append(make_tetrode(29, [84,85,86,87], position=[-4650, -3300]))
    tetrodes.append(make_tetrode(30, [76,77,78,79], position=[-4650, -3700]))
    tetrodes.append(make_tetrode(31, [80,81,82,83], position=[-4650, -3700]))

    layout = ProbeGroup()
    for tet in tetrodes:
        layout.add_probe(tet)
    return layout


def vlad_tetrodes():
    """ Generate layout for Vlad's drive with only tetrodes (12x2 bilaterally).
        Channel layout is for Intan RHD 64-channel headstage.
    """
    tetrodes = []
    tetrodes.append(make_tetrode(1, [48,49,50,51], position=[-3600, 1900]))
    tetrodes.append(make_tetrode(2, [56,57,58,59], position=[-3950, 2100]))
    tetrodes.append(make_tetrode(3, [44,45,46,47], position=[-3600, 2300]))

    tetrodes.append(make_tetrode(4, [36,37,38,39], position=[-3600, 3100]))
    tetrodes.append(make_tetrode(5, [70,71,72,73], position=[-3950, 2900]))
    tetrodes.append(make_tetrode(6, [66,67,76,77], position=[-3600, 2700]))

    tetrodes.append(make_tetrode(7, [52,53,54,55], position=[-3950, 3300]))
    tetrodes.append(make_tetrode(8, [60,61,62,63], position=[-4300, 3500]))
    tetrodes.append(make_tetrode(9, [40,41,42,43], position=[-4650, 3100]))

    tetrodes.append(make_tetrode(10, [32,33,34,35], position=[-4650, 3500]))
    tetrodes.append(make_tetrode(11, [68,69,74,75], position=[-4650, 3900]))
    tetrodes.append(make_tetrode(12, [64,65,78,79], position=[-4300, 3100]))

    tetrodes.append(make_tetrode(13, [12,13,14,15], position=[-4650, 3500]))
    tetrodes.append(make_tetrode(14, [4,5,6,7], position=[-4300, 2700]))
    tetrodes.append(make_tetrode(15, [16,17,18,19], position=[-4300, 2300]))

    tetrodes.append(make_tetrode(16, [24,25,26,27], position=[-3600, -1900]))
    tetrodes.append(make_tetrode(17, [86,87,88,89], position=[-3600, -2300]))
    tetrodes.append(make_tetrode(18, [82,83,92,93], position=[-3600, -3100]))

    tetrodes.append(make_tetrode(19, [8,9,10,11], position=[-3600, -3100]))
    tetrodes.append(make_tetrode(20, [0,1,2,3], position=[-3950, -2100]))
    tetrodes.append(make_tetrode(21, [20,21,22,23], position=[-3950, -2500]))

    tetrodes.append(make_tetrode(22, [28,29,30,31], position=[-3950, -2900]))
    tetrodes.append(make_tetrode(23, [90,91,84,85], position=[-3950, -3300]))
    tetrodes.append(make_tetrode(24, [80,81,94,95], position=[-4300, -2300]))

    layout = ProbeGroup()
    for tet in tetrodes:
        layout.add_probe(tet)
    return layout


## use this two vars to predefine implants
__def_implants_funcs = {"small_drive_rhd_64": small_drive_rhd_64,
                        "small_drive_og_rhd_64": small_drive_og_rhd_64,
                        "big_drive_og_rhd_64": big_drive_og_rhd_64,
                        "igor_drive_og_rhd_64": igor_drive_og_rhd_64,
                        "vlad_tetrodes": vlad_tetrodes}
DEFINED_IMPLANTS = list(__def_implants_funcs.keys())

def get_implant(name):
    if name not in DEFINED_IMPLANTS:
        raise ValueError(f"Implant {name} is not defined.")

    return __def_implants_funcs[name]()


def plot_implant_layout(layout, path=None):
    """ Plot given implant layout.

        Args:
            layout - implant layout to be plotted
            path - if given figure will be saved to this location\
                    if not (default) the figure will be displayed
    """
    plot_probe_group(layout)
    for tet in layout.probes:
        pos = np.mean(tet.contact_positions, axis=0)
        plt.text(*pos, tet.annotations["name"], ha='center', va='center', fontweight="demibold", fontsize=9)
        for cp, ci in zip(tet.contact_positions, tet.device_channel_indices):
            plt.text(*cp, str(ci), ha='center', va='center', fontsize=7, color="white")
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


def __write_layout_txt(path, layout):
    with open(path, "w") as f:
        num_tetrodes = len(layout.probes)
        f.write(f"{num_tetrodes}\n")
        for tet in layout.probes:
            channels = " ".join(list(map(str, tet.device_channel_indices)))
            f.write(f"{channels}\n")

        for tet in layout.probes:
            contacts = " ".join(list(map(str, tet.contact_positions.flatten())))
            f.write(f"{contacts}\n")


def write_layout(path, layout):
    if path.endswith(".prb"):
        write_prb(path, layout)
        return

    if path.endswith(".json"):
        write_probeinterface(path, layout)
        return

    if path.endswith(".txt"):
        __write_layout_txt(path, layout)
        return

    raise ValueError("Invalid path: supported extensions are '.prb', '.json' and '.txt'.")


def __read_layout_txt(path):
    """ Read tetrode drive layout from a '.txt' file.

        File should be in the following format:
            num_tetrodes
            for tet in tetrodes:
            channels...(4 integers)
            optional:
            for tet in tetrodes:
            contacts...(4 pairs (AP,ML) of coordinates, 8 numbers)

            Example for two detrodes without specified contacts(default will be used):
            2
            0 1 2 3
            4 5 6 7

            Example for two detrodes with specified contacts:
            2
            0 1 2 3
            4 5 6 7
            0 0 -20 20 20 20 0 40
            500 0 480 20 520 20 500 40

        Args:
            path - path to the '.txt' file with layout specification
        Return:
            layout (ProbeGroup)
    """
    with open(path, "r") as f:
        lines = f.readlines()
        try:
            num_tetrodes = int(lines[0].strip().split(' ')[0])
        except ValueError:  # make it more informative
            raise ValueError("Invalid content: number of tetrodes (the first line) must be an integer.")
        if len(lines) != num_tetrodes + 1 and len(lines) != (num_tetrodes * 2 + 1):
            raise ValueError(f"Invalid content: \
given file has {len(lines)} lines, but a \
'.txt' file with {num_tetrodes} tetrodes \
must have {num_tetrodes + 1} or {2 * num_tetrodes + 1} lines.")

        tetrodes = []
        channels = []
        for line in lines[1:num_tetrodes + 1]:
            lc = list(map(int, line.strip().split()))[-4:]
            if len(lc) != 4:
                raise ValueError(f"Invalid content in '.txt' file:\
each tetrode must have 4 channels.")
            channels.append(lc)

        if len(lines) == num_tetrodes * 2 + 1:
            # file has contacts
            contacts = []
            for line in lines[num_tetrodes+1:]:
                lc = map(int, line.strip().split())
                lc = np.fromiter(lc, float)
                if len(lc) != 8:
                    raise ValueError(f"Invalid content in '.txt' file:\
each tetrode must have 0 or 8 contact coordinates.")
                lc = lc.reshape(4,2)
                contacts.append(lc)
            
            for tn in range(num_tetrodes):
                tetrodes.append(make_tetrode(tn, channels[tn], contacts=contacts[tn]))
        else:
            # contacts are not specified in the file
            # use default
            for tn in range(num_tetrodes):
                tetrodes.append(make_tetrode(tn, channels[tn]))

        layout = ProbeGroup()
        [layout.add_probe(tet) for tet in tetrodes]
        return layout


def read_layout(path):
    if path.endswith(".prb"):
        return read_prb(path)

    if path.endswith(".json"):
        return read_probeinterface(path)

    if path.endswith(".txt"):
        return __read_layout_txt(path)

    raise ValueError("Invalid path: supported extensions are '.prb', '.json' and '.txt'.")
