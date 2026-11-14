"""Temporal snapshots, feature construction, and binary ranking targets."""

from typing import Tuple

import pandas as pd


_EVENT_TYPES = ("view", "addtocart", "transaction")


def split_history_target(
    events: pd.DataFrame,
    history_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split an existing training period by distinct timestamps."""
    if not 0 < history_fraction < 1:
        raise ValueError("history_fraction must be between 0 and 1.")
    if "event_time" not in events:
        raise ValueError("events must contain event_time.")
    unique_times = pd.Index(events["event_time"].drop_duplicates()).sort_values()
    if len(unique_times) < 2:
        raise ValueError("At least two distinct timestamps are required.")
    boundary_index = min(max(int(len(unique_times) * history_fraction), 1),
                         len(unique_times) - 1)
    target_start = pd.Timestamp(unique_times[boundary_index])
    history = events[events["event_time"] < target_start].copy()
    target = events[events["event_time"] >= target_start].copy()
    if history.empty or target.empty:
        raise ValueError("Both history and target windows must contain events.")
    if not history["event_time"].max() < target["event_time"].min():
        raise ValueError("History and target timestamps must be strictly ordered.")
    return history.reset_index(drop=True), target.reset_index(drop=True), target_start


def build_ranking_features(
    history_events: pd.DataFrame,
    candidates: pd.DataFrame,
    as_of: pd.Timestamp,
    snapshot_id: str,
) -> pd.DataFrame:
    """Calculate user, item, and candidate features from pre-snapshot history."""
    required_history = {"visitorid", "itemid", "event", "event_time"}
    missing_history = required_history - set(history_events.columns)
    if missing_history:
        raise ValueError(
            "history_events is missing columns: {}".format(sorted(missing_history))
        )
    required_candidates = {"user_id", "item_id", "als_score", "als_rank",
                           "has_als", "itemknn_score", "itemknn_rank",
                           "has_itemknn", "candidate_source"}
    missing_candidates = required_candidates - set(candidates.columns)
    if missing_candidates:
        raise ValueError(
            "candidates is missing columns: {}".format(sorted(missing_candidates))
        )
    if candidates.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Candidates must be unique by user_id and item_id.")

    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    event_time = pd.to_datetime(history_events["event_time"], utc=True, errors="coerce")
    if event_time.isna().any():
        raise ValueError("history_events contains invalid event_time values.")
    if (event_time >= cutoff).any():
        raise ValueError("history_events contains events at or after the snapshot.")

    history = history_events.copy()
    history["event_time"] = event_time
    user_event_counts = (
        history.groupby(["visitorid", "event"]).size().unstack(fill_value=0)
    )
    user_features = pd.DataFrame(index=user_event_counts.index)
    for event_type in _EVENT_TYPES:
        user_features["user_{}_count".format(event_type)] = (
            user_event_counts[event_type]
            if event_type in user_event_counts
            else 0
        )
    user_features["user_unique_items"] = history.groupby("visitorid")["itemid"].nunique()
    user_features["user_last_activity_hours"] = (
        (cutoff - history.groupby("visitorid")["event_time"].max())
        .dt.total_seconds()
        / 3600.0
    )
    user_features["user_total_events"] = user_features[
        ["user_{}_count".format(event_type) for event_type in _EVENT_TYPES]
    ].sum(axis=1)

    item_event_counts = history.groupby(["itemid", "event"]).size().unstack(fill_value=0)
    item_features = pd.DataFrame(index=item_event_counts.index)
    for event_type in _EVENT_TYPES:
        item_features["item_{}_count".format(event_type)] = (
            item_event_counts[event_type]
            if event_type in item_event_counts
            else 0
        )
    item_features["item_unique_users"] = history.groupby("itemid")["visitorid"].nunique()

    recent = history[history["event_time"] >= cutoff - pd.Timedelta(days=7)]
    item_features["item_event_count_7d"] = recent.groupby("itemid").size()
    item_features = item_features.fillna(0)

    result = candidates.copy()
    result["snapshot_id"] = str(snapshot_id)
    user_stats = user_features.rename_axis("user_id").reset_index()
    item_stats = item_features.rename_axis("item_id").reset_index()
    result = result.merge(user_stats, on="user_id", how="left", validate="many_to_one")
    result = result.merge(item_stats, on="item_id", how="left", validate="many_to_one")
    result[user_features.columns] = result[user_features.columns].fillna(0)
    result[item_features.columns] = result[item_features.columns].fillna(0)
    result["has_als"] = result["has_als"].astype(bool)
    result["has_itemknn"] = result["has_itemknn"].astype(bool)
    return result


def attach_binary_targets(
    features: pd.DataFrame,
    target_events: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Label candidate pairs from interactions strictly after the snapshot."""
    required = {"user_id", "item_id", "snapshot_id"}
    if not required.issubset(features.columns):
        raise ValueError("features must contain user_id, item_id, and snapshot_id.")
    if features.duplicated(["snapshot_id", "user_id", "item_id"]).any():
        raise ValueError("Each snapshot must have unique user-item candidate pairs.")
    if not {"visitorid", "itemid", "event_time"}.issubset(target_events.columns):
        raise ValueError("target_events must contain visitorid, itemid, and event_time.")

    cutoff = pd.Timestamp(as_of)
    cutoff = pd.Timestamp(cutoff.tz_localize("UTC") if cutoff.tzinfo is None
                          else cutoff.tz_convert("UTC"))
    target_times = pd.to_datetime(target_events["event_time"], utc=True, errors="coerce")
    if target_times.isna().any():
        raise ValueError("target_events contains invalid event_time values.")
    if (target_times < cutoff).any():
        raise ValueError("target_events contains events before the snapshot.")

    positive_pairs = set(
        zip(
            target_events["visitorid"].astype(int),
            target_events["itemid"].astype(int),
        )
    )
    result = features.copy()
    result["target"] = [
        int((int(user_id), int(item_id)) in positive_pairs)
        for user_id, item_id in zip(result["user_id"], result["item_id"])
    ]
    return result
