from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUTS = {
    "Clean": Path(
        "classifier_train_result/slice_model_train_result/"
        "eval_confidence_details.csv"
    ),
    "AdvTabDDPM": Path(
        "exp/colosseum/ddpm_tune_best/eval_confidence_details.csv"
    ),
}


def resolve_csv(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_dir():
        path = path / "eval_confidence_details.csv"
    return path


def load_metrics(name: str, csv_path: Path, delta: float) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"{name}: missing {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"true_label", "predicted_label", "confidence_difference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name}: {csv_path} is missing columns {sorted(missing)}"
        )

    true_label = df["true_label"].to_numpy()
    pred_label = df["predicted_label"].to_numpy()
    cd = df["confidence_difference"].to_numpy(dtype=float)

    correct = true_label == pred_label
    boundary = np.abs(cd) <= delta
    general = (~boundary) & correct
    out_of_boundary = (~boundary) & (~correct)

    metrics = {
        "dataset": name,
        "csv_path": str(csv_path),
        "samples": int(len(df)),
        "accuracy": float(correct.mean()),
        "error_rate": float((~correct).mean()),
        "mean_cd": float(cd.mean()),
        "median_cd": float(np.median(cd)),
        "boundary_ratio": float(boundary.mean()),
        "general_ratio": float(general.mean()),
        "out_of_boundary_ratio": float(out_of_boundary.mean()),
    }

    class_names = {0: "eMBB", 1: "mMTC", 2: "URLLC"}
    for class_id, class_name in class_names.items():
        mask = true_label == class_id
        metrics[f"{class_name}_accuracy"] = (
            float(correct[mask].mean()) if mask.any() else float("nan")
        )

    return metrics


def save_bar_chart(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
    percentage: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    if percentage:
        ax.set_ylim(0, max(1.0, max(values, default=1.0) * 1.15))
        for index, value in enumerate(values):
            ax.text(index, value, f"{value * 100:.2f}%", ha="center", va="bottom")
    else:
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.4f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_zone_chart(summary: pd.DataFrame, output_path: Path) -> None:
    labels = summary["dataset"].tolist()
    general = summary["general_ratio"].to_numpy()
    boundary = summary["boundary_ratio"].to_numpy()
    out_of_boundary = summary["out_of_boundary_ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, general, label="General")
    ax.bar(labels, boundary, bottom=general, label="Boundary")
    ax.bar(
        labels,
        out_of_boundary,
        bottom=general + boundary,
        label="Out of boundary",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sample ratio")
    ax.set_title("Decision-region composition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare clean and adversarial classifier evaluation CSVs."
    )
    parser.add_argument(
        "--clean",
        default=str(DEFAULT_INPUTS["Clean"]),
        help="Clean evaluation directory or eval_confidence_details.csv",
    )
    parser.add_argument(
        "--advtabddpm",
        default=str(DEFAULT_INPUTS["AdvTabDDPM"]),
        help="AdvTabDDPM evaluation directory or CSV",
    )
    parser.add_argument("--fgsm", default=None)
    parser.add_argument("--pgd", default=None)
    parser.add_argument("--cw", default=None)
    parser.add_argument(
        "--output-dir",
        default="results/attack_comparison",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.35,
        help="Boundary threshold used for |CD| <= delta.",
    )
    args = parser.parse_args()

    requested = [
        ("Clean", resolve_csv(args.clean)),
        ("AdvTabDDPM", resolve_csv(args.advtabddpm)),
        ("FGSM", resolve_csv(args.fgsm)),
        ("PGD", resolve_csv(args.pgd)),
        ("C&W", resolve_csv(args.cw)),
    ]

    rows = []
    for name, path in requested:
        if path is None:
            continue
        if not path.exists() and name not in {"Clean", "AdvTabDDPM"}:
            print(f"[Skip] {name}: {path} does not exist")
            continue
        rows.append(load_metrics(name, path, args.delta))

    if not rows:
        raise RuntimeError("No evaluation CSV files were loaded.")

    summary = pd.DataFrame(rows)

    clean_rows = summary[summary["dataset"] == "Clean"]
    if clean_rows.empty:
        raise RuntimeError("A clean result is required for accuracy comparison.")

    clean_accuracy = float(clean_rows.iloc[0]["accuracy"])
    summary["accuracy_drop_vs_clean"] = clean_accuracy - summary["accuracy"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "attack_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    labels = summary["dataset"].tolist()

    save_bar_chart(
        labels,
        summary["accuracy"].tolist(),
        "Accuracy",
        "Classifier accuracy: clean vs adversarial data",
        output_dir / "accuracy_comparison.png",
    )

    save_bar_chart(
        labels,
        summary["accuracy_drop_vs_clean"].tolist(),
        "Accuracy difference vs clean",
        "Classifier accuracy difference relative to clean baseline",
        output_dir / "accuracy_drop_comparison.png",
    )

    save_bar_chart(
        labels,
        summary["mean_cd"].tolist(),
        "Mean confidence difference",
        "Mean confidence difference by dataset",
        output_dir / "mean_cd_comparison.png",
        percentage=False,
    )

    save_zone_chart(
        summary,
        output_dir / "zone_ratio_comparison.png",
    )

    json_path = output_dir / "attack_comparison_summary.json"
    json_path.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    printable = summary.copy()
    percentage_columns = [
        "accuracy",
        "error_rate",
        "boundary_ratio",
        "general_ratio",
        "out_of_boundary_ratio",
        "accuracy_drop_vs_clean",
        "eMBB_accuracy",
        "mMTC_accuracy",
        "URLLC_accuracy",
    ]
    for column in percentage_columns:
        if column in printable:
            printable[column] = printable[column].map(
                lambda value: f"{value * 100:.2f}%"
                if pd.notna(value)
                else "N/A"
            )

    print(printable.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved figures: {output_dir}")


if __name__ == "__main__":
    main()
