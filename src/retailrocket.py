"""Retailrocket event loading and leakage-safe temporal splitting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "visitorid", "itemid", "event")


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological train, validation, and test event frames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_start: pd.Timestamp
    test_start: pd.Timestamp


def load_events(path: Path) -> pd.DataFrame:
    """Load and validate Retailrocket events; convert epoch milliseconds to UTC."""
    events = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(events.columns))
    if missing_columns:
        raise ValueError(
            "Events CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if events.empty:
        raise ValueError("Events CSV contains no rows.")

    for column in ("visitorid", "itemid", "event"):
        if events[column].isna().any():
            raise ValueError("Events CSV contains missing values in '{}'.".format(column))

    timestamps = pd.to_numeric(events["timestamp"], errors="coerce")
    event_time = pd.to_datetime(timestamps, unit="ms", utc=True, errors="coerce")
    if event_time.isna().any():
        raise ValueError(
            "Events CSV contains missing or invalid epoch-millisecond timestamps."
        )

    events = events.copy()
    events["event_time"] = event_time
    return events.sort_values("event_time", kind="mergesort").reset_index(drop=True)


def load_processed_events(path: Path) -> pd.DataFrame:
    """Load a generated split and normalize serialized timestamps to UTC."""
    events = pd.read_csv(path)
    if "event_time" not in events.columns:
        raise ValueError("Processed events CSV is missing the 'event_time' column.")
    event_time = pd.to_datetime(
        events["event_time"], utc=True, errors="coerce", format="mixed"
    )
    if event_time.isna().any():
        raise ValueError("Processed events CSV contains invalid event_time values.")
    events["event_time"] = event_time
    return events


def split_chronologically(
    events: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split by distinct event timestamps so simultaneous events never straddle sets.

    Fractions are applied to the ordered distinct timestamps, not individual rows.
    Therefore event counts can differ from the requested proportions when many
    events share a timestamp.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")
    if events.empty:
        raise ValueError("Cannot split an empty events frame.")
    if "event_time" not in events.columns:
        raise ValueError("Events frame must contain the parsed 'event_time' column.")

    unique_times = pd.Index(events["event_time"].drop_duplicates()).sort_values()
    timestamp_count = len(unique_times)
    if timestamp_count < 3:
        raise ValueError("At least three distinct timestamps are required for splitting.")

    validation_index = min(
        max(int(timestamp_count * train_fraction), 1),
        timestamp_count - 2,
    )
    test_index = min(
        max(int(timestamp_count * (train_fraction + validation_fraction)),
            validation_index + 1),
        timestamp_count - 1,
    )
    validation_start = unique_times[validation_index]
    test_start = unique_times[test_index]

    train = events.loc[events["event_time"] < validation_start].copy()
    validation = events.loc[
        (events["event_time"] >= validation_start)
        & (events["event_time"] < test_start)
    ].copy()
    test = events.loc[events["event_time"] >= test_start].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Each temporal split must contain at least one event.")

    split = TemporalSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        validation_start=validation_start,
        test_start=test_start,
    )
    validate_temporal_split(split)
    return split


def validate_temporal_split(split: TemporalSplit) -> None:
    """Raise when split boundaries overlap or are not strictly chronological."""
    if split.train.empty or split.validation.empty or split.test.empty:
        raise ValueError("Each temporal split must contain at least one event.")
    if not split.train["event_time"].max() < split.validation["event_time"].min():
        raise ValueError("Train and validation event times overlap or are out of order.")
    if not split.validation["event_time"].max() < split.test["event_time"].min():
        raise ValueError("Validation and test event times overlap or are out of order.")
    if split.validation_start != split.validation["event_time"].min():
        raise ValueError("validation_start must equal the first validation timestamp.")
    if split.test_start != split.test["event_time"].min():
        raise ValueError("test_start must equal the first test timestamp.")


def split_event_counts(split: TemporalSplit) -> Tuple[int, int, int]:
    """Return row counts in train, validation, and test order."""
    return len(split.train), len(split.validation), len(split.test)
