"""Quick viewer for an fMRI BOLD scan (exploratory, not part of the EEG pipeline).

The fMRI dataset (ds003126) is not included in this repository. Pass the path to
a .nii.gz volume:

    python view_fmri.py path/to/sub-XXX_task-read_run-01_bold.nii.gz

Requires nilearn and nibabel (see requirements.txt).
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scan', help='path to a .nii.gz fMRI volume')
    parser.add_argument('--save', metavar='OUT',
                        help='save the figure to this path instead of displaying it')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.scan):
        print(f"Scan not found: {args.scan}")
        return 1

    # Imported here so --help works without the optional neuroimaging deps installed.
    import matplotlib.pyplot as plt
    from nilearn import plotting

    print(f"Loading fMRI data from {args.scan}...")
    plotting.plot_epi(args.scan, title=os.path.basename(args.scan))

    if args.save:
        plt.savefig(args.save)
        print(f"Saved → {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
