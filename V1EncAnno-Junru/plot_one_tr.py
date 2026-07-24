"""Plot one fMRI volume (TR) from subject 01, movie run 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import image, plotting


BOLD_PATH = Path(
    "ds000113/sub-01/ses-movie/func/"
    "sub-01_ses-movie_task-movie_run-1_bold.nii.gz"
)
OUTPUT_PATH = Path("sub-01_run-1_TR-225.png")
TR_INDEX = 225  # Zero-based index: the 226th acquired volume.


def main() -> None:
    bold_img = nib.load(BOLD_PATH)
    if len(bold_img.shape) != 4:
        raise ValueError(f"Expected a 4D BOLD image, got shape {bold_img.shape}")
    if not 0 <= TR_INDEX < bold_img.shape[3]:
        raise IndexError(
            f"TR_INDEX={TR_INDEX} is outside the valid range "
            f"0..{bold_img.shape[3] - 1}"
        )

    one_tr = image.index_img(bold_img, TR_INDEX)
    tr_seconds = bold_img.header.get_zooms()[3]

    display = plotting.plot_epi(
        one_tr,
        display_mode="ortho",
        black_bg=True,
        annotate=True,
        draw_cross=False,
        colorbar=True,
        title=(
            f"Subject 01 | Movie run 1 | TR {TR_INDEX} "
            f"(t ≈ {TR_INDEX * tr_seconds:.1f} s)"
        ),
    )
    display.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    display.close()
    plt.close("all")

    print(f"BOLD shape: {bold_img.shape}")
    print(f"Header TR: {tr_seconds:.3f} s")
    print(f"Plotted zero-based TR index: {TR_INDEX}")
    print(f"Saved plot: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
