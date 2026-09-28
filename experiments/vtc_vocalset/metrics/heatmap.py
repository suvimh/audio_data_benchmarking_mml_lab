import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================

CSV_PATH = "vtc_vocalset_full_full_metrics_outline.csv"

OUTPUT_BACC = "supplementary_heatmaps_bacc.pdf"
OUTPUT_F1 = "supplementary_heatmaps_f1.pdf"

OUTPUT_BACC_PNG = "supplementary_heatmaps_bacc.png"
OUTPUT_F1_PNG = "supplementary_heatmaps_f1.png"


# Ordering
CLASSIFIER_ORDER = ["knn", "rf", "mlp", "svm"]
REPRESENTATION_ORDER = ["mfcc", "opensmile", "vggish", "whisper", "clap"]

CLASSIFIER_LABELS = {
    "knn": "KNN",
    "rf": "Random Forest",
    "mlp": "MLP",
    "svm": "SVM",
}

REPRESENTATION_LABELS = {
    "mfcc": "MFCC",
    "opensmile": "OpenSMILE",
    "vggish": "VGGish",
    "whisper": "Whisper",
    "clap": "CLAP",
}


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()


# ============================================================
# Traditional ML heatmap
# ============================================================


def make_traditional_matrix(metric):

    traditional = df[
        df["Model"].isin(CLASSIFIER_ORDER) & df["Input Data"].isin(REPRESENTATION_ORDER)
    ].copy()

    matrix = traditional.pivot(
        index="Model",
        columns="Input Data",
        values=metric,
    )

    matrix = matrix.reindex(
        index=CLASSIFIER_ORDER,
        columns=REPRESENTATION_ORDER,
    )

    matrix.index = [CLASSIFIER_LABELS[x] for x in matrix.index]

    matrix.columns = [REPRESENTATION_LABELS[x] for x in matrix.columns]

    return matrix


# ============================================================
# CLAP / Whisper fine-tuning heatmap
# ============================================================


def make_finetuning_matrix(metric):

    # Map CSV model names to readable backbone/training labels
    pretrained = df[
        df["Model"].isin(
            [
                "clap_head_only",
                "whisper_head_only",
                "clap_full",
                "whisper_full",
            ]
        )
    ].copy()

    # Determine backbone
    pretrained["Backbone"] = pretrained["Model"].map(
        {
            "clap_head_only": "CLAP",
            "clap_full": "CLAP",
            "whisper_head_only": "Whisper",
            "whisper_full": "Whisper",
        }
    )

    # Determine training strategy
    pretrained["Training"] = pretrained["Model"].map(
        {
            "clap_head_only": "Frozen encoder",
            "clap_full": "Fine-tuned encoder",
            "whisper_head_only": "Frozen encoder",
            "whisper_full": "Fine-tuned encoder",
        }
    )

    matrix = pretrained.pivot(
        index="Backbone",
        columns="Training",
        values=metric,
    )

    matrix = matrix.reindex(
        index=["CLAP", "Whisper"],
        columns=["Frozen encoder", "Fine-tuned encoder"],
    )

    return matrix


# ============================================================
# Plot function
# ============================================================


def make_figure(metric, output_pdf, output_png):

    traditional_matrix = make_traditional_matrix(metric)
    finetuning_matrix = make_finetuning_matrix(metric)

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.0),
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )

    # --------------------------------------------------------
    # Panel (a): Classifier × representation
    # --------------------------------------------------------

    sns.heatmap(
        traditional_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot_kws={
            "fontsize": 7,
        },
        ax=axes[0],
    )

    axes[0].set_xlabel("Representation", fontsize=9)
    axes[0].set_ylabel("Classifier", fontsize=9)

    axes[0].tick_params(
        axis="x",
        labelrotation=0,
        labelsize=8,
    )

    axes[0].tick_params(
        axis="y",
        labelrotation=0,
        labelsize=8,
    )

    axes[0].set_title(
        "(a) Classifier × Representation",
        fontsize=9,
        pad=8,
    )

    # --------------------------------------------------------
    # Panel (b): Backbone × training strategy
    # --------------------------------------------------------

    sns.heatmap(
        finetuning_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        annot_kws={
            "fontsize": 7,
        },
        ax=axes[1],
    )

    axes[1].set_xlabel("Training strategy", fontsize=9)
    axes[1].set_ylabel("Backbone", fontsize=9)

    axes[1].tick_params(
        axis="x",
        labelrotation=25,
        labelsize=7,
    )

    axes[1].tick_params(
        axis="y",
        labelrotation=0,
        labelsize=8,
    )

    axes[1].set_title(
        "(b) Backbone × Training Strategy",
        fontsize=9,
        pad=8,
    )

    # --------------------------------------------------------
    # Shared colorbar
    # --------------------------------------------------------

    # Create a ScalarMappable purely for the shared colorbar
    import matplotlib as mpl

    norm = mpl.colors.Normalize(
        vmin=0,
        vmax=1,
    )

    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap="viridis",
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        fraction=0.025,
        pad=0.03,
    )

    cbar.set_label(
        metric,
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    # --------------------------------------------------------
    # Overall layout
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.20,
        top=0.84,
        wspace=0.45,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)


# ============================================================
# Generate figures
# ============================================================

make_figure(
    metric="Val Balanced Accuracy",
    output_pdf=OUTPUT_BACC,
    output_png=OUTPUT_BACC_PNG,
)

make_figure(
    metric="Val F1",
    output_pdf=OUTPUT_F1,
    output_png=OUTPUT_F1_PNG,
)
