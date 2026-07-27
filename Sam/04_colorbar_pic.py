import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn.plotting import cm

# short script just for the colorbar; very minor

experiment = "localizers"
vmax = 12

fig_cb, ax_cb = plt.subplots(figsize=(1.2, 6))
norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
cb = mpl.colorbar.Colorbar(ax_cb, cmap="cold_white_hot", norm=norm, orientation="vertical")
cb.set_label("z-value")
fig_cb.savefig(f"colorbar_{experiment}_cold_white_hot.jpg", dpi=200, bbox_inches="tight")
