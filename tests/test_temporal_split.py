import pandas as pd
import pytest

from scripts.run_eda import create_eda_outputs, write_temporal_splits
from src.retailrocket import load_events, load_processed_events, split_chronologically


def _events_at(*timestamps):
    return pd.DataFrame(
        {
            "event_time": pd.to_datetime(timestamps, unit="ms", utc=True),
            "visitorid": list(range(len(timestamps))),
            "itemid": list(range(len(timestamps))),
            "event": ["view"] * len(timestamps),
        }
    )


def test_split_keeps_equal_timestamps_together_and_strictly_ordered():
    events = _events_at(1000, 1000, 2000, 3000, 4000, 5000, 6000, 7000)

    split = split_chronologically(events, train_fraction=0.5, validation_fraction=0.25)

    assert split.train["event_time"].max() < split.validation["event_time"].min()
    assert split.validation["event_time"].max() < split.test["event_time"].min()
    assert not set(split.train["event_time"]) & set(split.validation["event_time"])
    assert not set(split.validation["event_time"]) & set(split.test["event_time"])
    assert len(split.train) + len(split.validation) + len(split.test) == len(events)


def test_split_rejects_too_few_distinct_timestamps():
    events = _events_at(1000, 1000, 2000)

    with pytest.raises(ValueError, match="three distinct timestamps"):
        split_chronologically(events)


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [(0, 0.2), (0.8, 0.2), (0.6, -0.1)],
)
def test_split_rejects_invalid_fractions(train_fraction, validation_fraction):
    events = _events_at(1000, 2000, 3000, 4000)

    with pytest.raises(ValueError):
        split_chronologically(events, train_fraction, validation_fraction)


def test_loader_converts_epoch_milliseconds_and_checks_schema(tmp_path):
    events_path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "timestamp": [1000, 2000],
            "visitorid": [10, 11],
            "itemid": [20, 21],
            "event": ["view", "transaction"],
        }
    ).to_csv(events_path, index=False)

    loaded = load_events(events_path)

    assert str(loaded["event_time"].dtype) == "datetime64[ns, UTC]"
    assert loaded["event_time"].is_monotonic_increasing
    assert loaded["event_time"].iloc[0] == pd.Timestamp("1970-01-01T00:00:01Z")

    pd.DataFrame({"timestamp": [1]}).to_csv(events_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_events(events_path)


def test_processed_loader_normalizes_mixed_iso_timestamps_to_utc(tmp_path):
    split_path = tmp_path / "train.csv"
    split_path.write_text(
        "event_time,visitorid,itemid,event\n"
        "2015-05-03 03:00:04.384000+00:00,1,10,view\n"
        "2015-05-03 03:00:11+00:00,2,20,transaction\n"
    )

    loaded = load_processed_events(split_path)

    assert str(loaded["event_time"].dtype) == "datetime64[ns, UTC]"
    assert loaded["event_time"].is_monotonic_increasing


def test_eda_and_split_outputs_are_written_from_events(tmp_path):
    events = _events_at(1000, 1000, 2000, 3000, 4000, 5000, 6000, 7000)
    events["event"] = ["view", "view", "addtocart", "view", "transaction",
                       "view", "addtocart", "transaction"]

    summary = create_eda_outputs(events, tmp_path / "outputs")
    manifest = write_temporal_splits(events, tmp_path / "processed")

    assert summary["event_rows"] == len(events)
    assert summary["event_counts"] == {
        "addtocart": 2,
        "transaction": 2,
        "view": 4,
    }
    split_frames = {
        name: pd.read_csv(tmp_path / "processed" / "{}.csv".format(name))
        for name in ("train", "validation", "test")
    }
    assert manifest["event_counts"] == {
        name: len(frame) for name, frame in split_frames.items()
    }
    assert sum(manifest["event_counts"].values()) == len(events)
    train_end = pd.to_datetime(split_frames["train"]["event_time"], utc=True).max()
    validation_times = pd.to_datetime(
        split_frames["validation"]["event_time"], utc=True
    )
    test_start = pd.to_datetime(split_frames["test"]["event_time"], utc=True).min()
    assert train_end < validation_times.min()
    assert validation_times.max() < test_start
    assert (tmp_path / "outputs" / "metrics" / "eda_summary.json").is_file()
    assert len(list((tmp_path / "outputs" / "figures").glob("*.png"))) == 4
    assert (tmp_path / "processed" / "train.csv").is_file()
    assert (tmp_path / "processed" / "validation.csv").is_file()
    assert (tmp_path / "processed" / "test.csv").is_file()
    assert (tmp_path / "processed" / "split_manifest.json").is_file()
