import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


standardsize = 50


# ==========================
# Load frozen data
# ==========================

data = pd.read_csv("spark_jct_final.csv")


fig = plt.figure(figsize=(30, 12))
ax = fig.add_subplot(1, 1, 1)


scheduler_order = [
    "YARN",
    "Toposch-n",
    "Quasar",
    "ROSE",
    "COLTER",
    "DRL-Sched",
    "PID-Sched"
]


color_palette = [
    "#4c40f5",
    "#2E8B57",
    "#87CEFA",
    "#FF8C00",
    "#A9A9A9",
    "#f7fa9d",
    "#E74C3C"
]


# ==========================
# Boxplot
# ==========================

sns.boxplot(
    data=data,
    x="workloads",
    y="jct",
    hue="scheduler",
    hue_order=scheduler_order,
    palette=color_palette,
    meanprops={
        "marker": "D",
        "markerfacecolor": "indianred"
    },
    medianprops={
        "linestyle": "--",
        "color": "red"
    },
    whiskerprops={
        "linestyle": "--"
    },
    whis=2,
    width=0.7,
    linewidth=1.5,
    ax=ax
)


# ==========================
# Scatter points
# ==========================

sns.stripplot(
    data=data,
    x="workloads",
    y="jct",
    hue="scheduler",
    hue_order=scheduler_order,
    dodge=True,
    alpha=0.45,
    size=4,
    palette=color_palette,
    ax=ax
)


# ==========================
# Remove duplicated legend
# ==========================

handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles[:len(scheduler_order)],
    labels[:len(scheduler_order)],
    bbox_to_anchor=(1.02, 1.0),
    loc="upper left",
    ncol=1,
    fontsize=standardsize-10,
    frameon=True,
    edgecolor="gray",
    prop={
        "family": "Arial",
        "size": standardsize
    }
)


# ==========================
# Labels
# ==========================

ax.set_ylabel(
    "Normalized JCT of Co-located \n batch jobs",
    fontsize=standardsize,
    fontname="Arial",
    labelpad=25
)

ax.set_xlabel(
    "",
    fontsize=standardsize,
    fontname="Arial"
)


ax.tick_params(
    labelsize=standardsize+10,
    color="k"
)


ax.grid(
    which="major",
    axis="y",
    linestyle="--",
    linewidth=1
)


plt.savefig(
    "enhanced_batch_jct_comparison_csv.pdf",
    bbox_inches="tight"
)


plt.show()