"""Generate Retailrocket EDA figures, summary, and chronological data splits."""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.retailrocket import load_events, split_chronologically, split_event_counts


ROOT = Path(__file__).resolve().parents[1]


def _save_event_type_plot(events: pd.DataFrame, output_path: Path) -> None:
    counts = events["event"].value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(8, 5))
    counts.plot(kind="barh", ax=axis, color="#4472C4")
    axis.set(title="Retailrocket events by type", xlabel="Event count", ylabel="")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_daily_activity_plot(events: pd.DataFrame, output_path: Path) -> None:
    daily = (
        events.assign(event_day=events["event_time"].dt.floor("D"))
        .groupby(["event_day", "event"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    daily.plot(ax=axis, linewidth=1.2)
    axis.set(
        title="Daily event volume",
        xlabel="Event date (UTC)",
        ylabel="Event count",
    )
    axis.legend(title="Event")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_top_items_plot(events: pd.DataFrame, output_path: Path) -> None:
    counts = events["itemid"].value_counts().head(20).sort_values()
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(counts.index.astype(str), counts.values, color="#70AD47")
    axis.set(title="Top 20 items by interaction count", xlabel="Event count", ylabel="Item ID")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_user_activity_plot(events: pd.DataFrame, output_path: Path) -> None:
    activity = events.groupby("visitorid").size()
    upper_edge = max(int(activity.max()) + 1, 2)
    bins = np.geomspace(1, upper_edge, num=30)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(activity, bins=bins, color="#ED7D31", edgecolor="white")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        title="Interactions per visitor",
        xlabel="Events per visitor (log scale)",
        ylabel="Visitors (log scale)",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def create_eda_outputs(events: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    """Write descriptive statistics and four event-level EDA visualizations."""
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "event_rows": int(len(events)),
        "unique_visitors": int(events["visitorid"].nunique()),
        "unique_items": int(events["itemid"].nunique()),
        "event_time_start_utc": events["event_time"].min().isoformat(),
        "event_time_end_utc": events["event_time"].max().isoformat(),
        "event_counts": {
            str(event): int(count)
            for event, count in events["event"].value_counts().sort_index().items()
        },
        "events_per_visitor": {
            "median": float(events.groupby("visitorid").size().median()),
            "p90": float(events.groupby("visitorid").size().quantile(0.90)),
            "max": int(events.groupby("visitorid").size().max()),
        },
    }
    with (metrics_dir / "eda_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    _save_event_type_plot(events, figures_dir / "event_types.png")
    _save_daily_activity_plot(events, figures_dir / "daily_activity.png")
    _save_top_items_plot(events, figures_dir / "top_items.png")
    _save_user_activity_plot(events, figures_dir / "user_activity.png")
    return summary


def write_temporal_splits(events: pd.DataFrame, processed_dir: Path) -> Dict[str, object]:
    split = split_chronologically(events)
    processed_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    for name, frame in frames.items():
        frame.to_csv(processed_dir / "{}.csv".format(name), index=False)

    train_count, validation_count, test_count = split_event_counts(split)
    manifest = {
        "method": "chronological split by distinct UTC event timestamps",
        "fractions_by_distinct_timestamp": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "validation_start_utc": split.validation_start.isoformat(),
        "test_start_utc": split.test_start.isoformat(),
        "event_counts": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        },
        "max_train_time_utc": split.train["event_time"].max().isoformat(),
        "min_validation_time_utc": split.validation["event_time"].min().isoformat(),
        "max_validation_time_utc": split.validation["event_time"].max().isoformat(),
        "min_test_time_utc": split.test["event_time"].min().isoformat(),
    }
    with (processed_dir / "split_manifest.json").open(
        "w", encoding="utf-8"
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "raw" / "events.csv",
        help="Path to Retailrocket events.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs",
        help="Directory for EDA figures and summary files.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            "Dataset not found at {}. Place Retailrocket events.csv at "
            "data/raw/events.csv or pass --input.".format(args.input)
        )

    events = load_events(args.input)
    summary = create_eda_outputs(events, args.output)
    manifest = write_temporal_splits(events, ROOT / "data" / "processed")
    print(
        "EDA completed for {event_rows} events and wrote chronological splits "
        "with counts {event_counts}.".format(
            event_rows=summary["event_rows"],
            event_counts=manifest["event_counts"],
        )
    )


if __name__ == "__main__":
    main()
