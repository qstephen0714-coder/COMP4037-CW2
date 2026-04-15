from pathlib import Path
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INPUT_FILE = Path("long_data.xlsx")
OUTPUT_FIG = Path("matrix_heatmap.png")

AGE_BAND_MAP = {
    "Age 0": "0-14",
    "Age 1-4": "0-14",
    "Age 5-9": "0-14",
    "Age 10-14": "0-14",
    "Age 15": "15-19",
    "Age 16": "15-19",
    "Age 17": "15-19",
    "Age 18": "15-19",
    "Age 19": "15-19",
    "Age 20-24": "20-29",
    "Age 25-29": "20-29",
    "Age 30-34": "30-39",
    "Age 35-39": "30-39",
    "Age 40-44": "40-49",
    "Age 45-49": "40-49",
    "Age 50-54": "50-59",
    "Age 55-59": "50-59",
    "Age 60-64": "60-69",
    "Age 65-69": "60-69",
    "Age 70-74": "70-79",
    "Age 75-79": "70-79",
    "Age 80-84": "80-89",
    "Age 85-89": "80-89",
    "Age 90+": "90+"
}

AGE_BAND_ORDER = ["0-14", "15-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]

LABEL_MAP = {
    "Other chronic obstructive pulmonary disease": "Chronic obstructive pulmonary disease",
    "Disorders of lipoprotein metabolism and other lipidaemias": "Lipid metabolism disorders",
    "Personal history of malignant neoplasm": "History of malignant neoplasm",
    "Chronic ischaemic heart disease": "Chronic ischaemic heart disease",
    "Diverticular disease of intestine": "Diverticular disease of intestine",
    "Presence of other functional implants": "Functional implants",
    "Atrial fibrillation and flutter": "Atrial fibrillation and flutter",
    "Chronic kidney disease": "Chronic kidney disease",
    "Heart failure": "Heart failure",
    "Other arthrosis": "Other arthrosis",
}


def wrap_label(text: str, width: int = 28) -> str:
    # Wrap long labels for display
    return "\n".join(textwrap.wrap(str(text), width=width))


def main() -> None:
    df = pd.read_excel(INPUT_FILE)

    # Aggregate age groups
    df = df[df["Age Group"].isin(AGE_BAND_MAP.keys())].copy()
    df["Age Band"] = df["Age Group"].map(AGE_BAND_MAP)

    band_df = (
        df.groupby(["Diagnosis Description", "All diagnoses", "Age Band"], as_index=False)["Admissions"]
        .sum()
    )
    band_df["Age Share"] = band_df["Admissions"] / band_df["All diagnoses"]

    # Select high-burden categories, then keep the most age-concentrated ones
    top25 = (
        band_df[["Diagnosis Description", "All diagnoses"]]
        .drop_duplicates()
        .sort_values("All diagnoses", ascending=False)
        .head(25)
    )

    candidate_df = band_df[band_df["Diagnosis Description"].isin(top25["Diagnosis Description"])].copy()

    concentration = (
        candidate_df.groupby("Diagnosis Description")["Age Share"]
        .max()
        .reset_index(name="PeakShare")
        .sort_values("PeakShare", ascending=False)
    )

    top10_names = concentration.head(10)["Diagnosis Description"].tolist()
    plot_df = candidate_df[candidate_df["Diagnosis Description"].isin(top10_names)].copy()

    plot_df["Age Band"] = pd.Categorical(plot_df["Age Band"], categories=AGE_BAND_ORDER, ordered=True)

    plot_df["Diagnosis Label"] = (
        plot_df["Diagnosis Description"]
        .map(LABEL_MAP)
        .fillna(plot_df["Diagnosis Description"])
    )

    # Order rows by peak age band
    peak_band = (
        plot_df.loc[
            plot_df.groupby("Diagnosis Label")["Age Share"].idxmax(),
            ["Diagnosis Label", "Age Band", "Age Share"]
        ]
        .rename(columns={"Age Band": "PeakBand"})
    )

    peak_band["PeakBand"] = pd.Categorical(peak_band["PeakBand"], categories=AGE_BAND_ORDER, ordered=True)
    peak_band = peak_band.sort_values(["PeakBand", "Age Share"], ascending=[True, False])

    ordered_labels = peak_band["Diagnosis Label"].tolist()

    wrapped_map = {label: wrap_label(label, 28) for label in ordered_labels}
    plot_df["Diagnosis Label Wrapped"] = plot_df["Diagnosis Label"].map(wrapped_map)
    ordered_wrapped_labels = [wrapped_map[label] for label in ordered_labels]

    heatmap_data = (
        plot_df.pivot_table(
            index="Diagnosis Label Wrapped",
            columns="Age Band",
            values="Age Share",
            aggfunc="sum"
        )
        .reindex(index=ordered_wrapped_labels, columns=AGE_BAND_ORDER)
    )

    custom_cmap = LinearSegmentedColormap.from_list(
        "elegant_age_map",
        [
            "#f7f3eb",
            "#e7ddd4",
            "#cdbfca",
            "#a58fb1",
            "#7b73a3",
            "#4f6792",
            "#244a73"
        ]
    )

    fig, ax = plt.subplots(figsize=(11.8, 6.8), facecolor="white")

    im = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        interpolation="nearest",
        cmap=custom_cmap
    )

    ax.set_title(
        "Selected admission categories peak at different later-life stages",
        fontsize=16,
        pad=16,
        weight="semibold"
    )
    ax.set_xlabel("Age band", fontsize=11)
    ax.set_ylabel("Diagnosis category", fontsize=11)

    ax.set_xticks(range(len(AGE_BAND_ORDER)))
    ax.set_xticklabels(AGE_BAND_ORDER, fontsize=10)

    ax.set_yticks(range(len(ordered_wrapped_labels)))
    ax.set_yticklabels(ordered_wrapped_labels, fontsize=10)

    ax.set_xticks([x - 0.5 for x in range(1, len(AGE_BAND_ORDER))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(ordered_wrapped_labels))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.025)
    cbar.set_label("Within-category age share", fontsize=10)
    cbar.outline.set_visible(False)

    # Show labels only for the largest values
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data.iloc[i, j]
            if pd.notna(val) and val >= 0.30:
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white",
                    weight="bold"
                )

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=450, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"Saved figure to: {OUTPUT_FIG}")

if __name__ == "__main__":
    main()