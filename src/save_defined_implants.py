import implants
from os.path import join


out_folder = "../implants"

for imp in implants.DEFINED_IMPLANTS:
    imp_layout = implants.get_implant(imp)
    implants.write_layout(join(out_folder, f"{imp}.txt"), imp_layout)
    implants.write_layout(join(out_folder, f"{imp}.prb"), imp_layout)
    implants.write_layout(join(out_folder, f"{imp}.json"), imp_layout)
    implants.plot_implant_layout(imp_layout, join(out_folder, f"{imp}.jpg"))
    print(f"Saved {imp} with {len(imp_layout.probes)} tetrodes.")
