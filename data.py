"""StatsBomb 무료 공개 데이터를 내려받고, PC(data 폴더)에 저장해 두는 모듈."""

import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

# statsbombpy는 무료 데이터를 쓸 때마다 "로그인 정보 없음" 경고를 띄운다. 무료 데이터만 쓰므로 숨긴다.
warnings.filterwarnings("ignore", module="statsbombpy")

DATA_DIR = Path(__file__).parent / "data"

# 앱에서 고를 수 있는 대회: 이름 -> (competition_id, season_id)
COMPETITIONS = {
    "FIFA 월드컵 2022": (43, 106),
    "UEFA 유로 2024": (55, 282),
    "코파 아메리카 2024": (223, 282),
    "분데스리가 2023/24": (9, 281),
    "리그1 2022/23": (7, 235),
    "라리가 2015/16": (11, 27),
    "프리미어리그 2015/16": (2, 27),
}

# 분석에 쓰는 열만 남겨서 저장 용량과 메모리를 줄인다.
EVENT_COLUMNS = [
    "match_id", "period", "minute", "second", "type", "team", "player", "player_id", "position",
    "location", "under_pressure", "play_pattern", "possession", "possession_team",
    "pass_end_location", "pass_outcome", "pass_type", "pass_shot_assist", "pass_goal_assist",
    "pass_length", "pass_height", "pass_cross", "pass_through_ball", "pass_switch",
    "carry_end_location",
    "shot_statsbomb_xg", "shot_outcome", "shot_type",
    "dribble_outcome", "duel_type", "duel_outcome", "interception_outcome",
    "foul_committed_card", "bad_behaviour_card",
    "substitution_replacement", "substitution_replacement_id",
    "tactics",
]


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """대회의 경기 목록."""
    return sb.matches(competition_id=competition_id, season_id=season_id)


def cache_path(competition_id: int, season_id: int) -> Path:
    return DATA_DIR / f"events_{competition_id}_{season_id}.pkl"


def is_cached(competition_id: int, season_id: int) -> bool:
    return cache_path(competition_id, season_id).exists()


def load_competition_events(competition_id: int, season_id: int, on_progress=None) -> pd.DataFrame:
    """대회 전체 경기의 이벤트(패스, 슈팅 등)를 불러온다.

    처음 한 번은 인터넷에서 내려받아 data 폴더에 저장하고, 그다음부터는 저장된 파일을 바로 읽는다.
    on_progress(완료한 경기 수, 전체 경기 수)를 넘기면 진행 상황을 알려준다.
    """
    path = cache_path(competition_id, season_id)
    if path.exists():
        return pd.read_pickle(path)

    matches = load_matches(competition_id, season_id)
    frames = []
    for i, match_id in enumerate(matches["match_id"], start=1):
        events = sb.events(match_id=match_id)
        frames.append(events[[c for c in EVENT_COLUMNS if c in events.columns]])
        if on_progress:
            on_progress(i, len(matches))

    all_events = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    all_events.to_pickle(path)
    return all_events
