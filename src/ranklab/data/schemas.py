from __future__ import annotations

from dataclasses import dataclass

INTERACTION_REQUIRED = (
    "user_id", "video_id", "date", "time_ms", "is_click", "is_like",
    "is_follow", "is_comment", "is_forward", "is_hate", "long_view",
    "play_time_ms", "duration_ms", "is_rand", "tab",
)
INTERACTION_RENAME = {"video_id": "item_id", "time_ms": "timestamp_ms", "is_rand": "is_random"}
LABEL_COLUMNS = {"is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "long_view", "play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter"}

@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    timestamp_ms: int
    date: int
    tab: int
    long_view: int
    is_click: int
    is_like: int
    is_follow: int
    is_comment: int
    is_forward: int
    is_hate: int
    play_time_ms: int
    duration_ms: int
    is_random: int
    source_log: str
