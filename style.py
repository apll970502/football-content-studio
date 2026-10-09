"""팀 전술 스타일을 경기 기록으로 계산하는 모듈. AI 없이 계산만으로 동작한다.

한 경기에서 한 팀이 어떻게 공격하고 수비했는지를 숫자(스타일 지표)로 바꾼다.
좌표는 StatsBomb 기준: 가로 120 x 세로 80, 모든 이벤트는 "그 행동을 한 팀이 오른쪽으로 공격"하는 방향.

스코어 상황 보정: 이기는 팀은 내려앉고 지는 팀은 몰아붙이기 때문에, 경기 전체 기록을 쓰면
"전술 때문에 결과가 났는지, 결과 때문에 전술이 바뀌었는지" 구분이 안 된다.
그래서 스타일과 xG는 **동점인 시간대**에서만 계산한다. 결정력(득점 - xG)만 경기 전체로 계산한다.
"""

import numpy as np
import pandas as pd

# 스타일 지표: 열 이름 -> (화면 이름, 설명). "횟수" 지표는 동점 시간 90분 기준으로 환산한다.
STYLE_FEATURES = {
    "possession": ("점유율(%)", "패스 중 우리 팀 패스 비율"),
    "ppda": ("PPDA(낮을수록 강한 압박)", "상대가 자기 진영에서 패스를 몇 번 할 때마다 우리가 태클·인터셉트·파울을 1번 하는지"),
    "def_height": ("수비 행동 높이(m)", "태클·인터셉트·압박 등 수비 행동의 평균 위치. 높을수록 앞에서 수비"),
    "recovery_height": ("볼 탈취 위치(m)", "공을 빼앗아 공격을 시작한 평균 위치. 높을수록 전방에서 뺏음"),
    "long_ball_share": ("롱볼 비율(%)", "자기 진영에서 시작한 패스 중 32m 이상 긴 패스 비율"),
    "pass_length": ("평균 패스 거리(m)", "짧은 패스 위주인지, 길게 차는지"),
    "crosses": ("크로스 수", "90분당 크로스 시도"),
    "wide_entry_share": ("측면 공격 비율(%)", "공격 3지역 진입 중 측면으로 들어간 비율"),
    "through_balls": ("스루패스 수", "90분당 수비 뒷공간으로 찌르는 패스"),
    "fast_attacks": ("빠른 전환 공격 수", "90분당, 공을 따낸 뒤 15초 안에 슈팅하거나 페널티박스에 들어간 공격"),
    "final_third_entries": ("공격 3지역 진입 수", "90분당 패스·드리블로 상대 진영 마지막 1/3에 들어간 횟수"),
    "box_entries_allowed": ("허용한 박스 진입 수", "90분당 상대가 우리 페널티박스로 공을 넣은 횟수. 적을수록 단단한 수비"),
    "opp_shot_distance": ("상대 슈팅 평균 거리(m)", "상대가 얼마나 먼 곳에서 슈팅했는지. 멀수록 좋은 위치를 안 내준 것"),
}
PER90_FEATURES = {"crosses", "through_balls", "fast_attacks", "final_third_entries", "box_entries_allowed"}

DEFENSIVE_TYPES = {"Pressure", "Interception", "Ball Recovery", "Block", "Clearance", "Foul Committed"}
PPDA_TYPES = {"Interception", "Foul Committed"}  # + 태클
WIN_OUTCOMES = {"Won", "Success In Play", "Success Out"}
YARD_TO_M = 0.9144
MIN_LEVEL_MINUTES = 15  # 동점 시간이 이보다 짧은 경기는 스타일 계산에서 뺀다


def _x(points: pd.Series) -> pd.Series:
    return points.str[0]


def _y(points: pd.Series) -> pd.Series:
    return points.str[1]


def _in_box(points: pd.Series) -> pd.Series:
    x, y = _x(points), _y(points)
    return (x >= 102) & (y >= 18) & (y <= 62)


def _seconds(ev: pd.DataFrame) -> pd.Series:
    return ev["minute"] * 60 + ev["second"]


def add_score_state(ev: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """각 이벤트 직전의 스코어 차이(홈 - 원정)를 붙인다."""
    ev = ev.sort_values(["period", "minute", "second"], kind="stable")
    scored = (ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal") | (ev["type"] == "Own Goal For")
    home_goal = (scored & (ev["team"] == home)).astype(int)
    away_goal = (scored & (ev["team"] == away)).astype(int)
    # 골 이벤트 자체는 골이 들어가기 "전" 상태로 본다
    diff = (home_goal.cumsum() - home_goal) - (away_goal.cumsum() - away_goal)
    return ev.assign(score_diff=diff)


def _fast_attacks(ev: pd.DataFrame, team: str) -> int:
    """공을 따낸 뒤 15초 안에 슈팅 또는 박스 진입까지 간 공격 수."""
    poss = ev[(ev["possession_team"] == team) & ev["play_pattern"].isin(["Regular Play", "From Counter"])]
    count = 0
    for _, p in poss.groupby("possession"):
        start = _seconds(p).iloc[0]
        quick = p[(_seconds(p) - start <= 15) & (p["team"] == team)]
        shot = (quick["type"] == "Shot").any()
        box = (quick["type"] == "Pass") & quick["pass_outcome"].isna() & _in_box(quick["pass_end_location"])
        box |= (quick["type"] == "Carry") & _in_box(quick["carry_end_location"])
        if shot or box.any():
            count += 1
    return count


def _box_entries(ev: pd.DataFrame, team: str) -> int:
    """팀이 상대 페널티박스로 공을 넣은 횟수 (박스 밖에서 안으로 들어간 패스 성공·드리블)."""
    mine = ev[ev["team"] == team]
    passes = mine[(mine["type"] == "Pass") & mine["pass_outcome"].isna()]
    carries = mine[mine["type"] == "Carry"]
    return int((~_in_box(passes["location"]) & _in_box(passes["pass_end_location"])).sum()
               + (~_in_box(carries["location"]) & _in_box(carries["carry_end_location"])).sum())


def _team_features(ev: pd.DataFrame, team: str, opponent: str, minutes: float) -> dict:
    """주어진 시간대(ev)에서 한 팀의 스타일 지표와 xG."""
    mine = ev[ev["team"] == team]
    theirs = ev[ev["team"] == opponent]
    my_passes = mine[mine["type"] == "Pass"]
    their_passes = theirs[theirs["type"] == "Pass"]
    tackles = (mine["type"] == "Duel") & (mine["duel_type"] == "Tackle")

    # 수비 높이 (압박 시도 포함)
    my_def = mine[mine["type"].isin(DEFENSIVE_TYPES) | tackles]
    # PPDA (표준 정의)
    their_build_up = their_passes[_x(their_passes["location"]) < 72]
    ppda_actions = mine[(mine["type"].isin(PPDA_TYPES) | tackles) & (_x(mine["location"]) > 48)]
    # 볼 탈취 위치: 공을 실제로 따낸 행동
    won = ((mine["type"] == "Ball Recovery")
           | ((mine["type"] == "Interception") & mine["interception_outcome"].isin(WIN_OUTCOMES))
           | (tackles & mine["duel_outcome"].isin(WIN_OUTCOMES)))
    # 빌드업
    own_half = my_passes[_x(my_passes["location"]) < 60]
    # 공격 3지역 진입
    ok_passes = my_passes[my_passes["pass_outcome"].isna()]
    carries = mine[mine["type"] == "Carry"]
    entries = pd.concat([
        ok_passes[(_x(ok_passes["location"]) < 80) & (_x(ok_passes["pass_end_location"]) >= 80)]["pass_end_location"],
        carries[(_x(carries["location"]) < 80) & (_x(carries["carry_end_location"]) >= 80)]["carry_end_location"],
    ])
    # 슈팅 (페널티 제외)
    my_shots = mine[(mine["type"] == "Shot") & (mine["shot_type"] != "Penalty")]
    their_shots = theirs[(theirs["type"] == "Shot") & (theirs["shot_type"] != "Penalty")]
    # 상대 슈팅 거리: 골대 중앙(120, 40)까지
    opp_dist = np.hypot(120 - _x(their_shots["location"]), 40 - _y(their_shots["location"])) * YARD_TO_M

    per90 = 90 / minutes
    return {
        "possession": len(my_passes) / max(len(my_passes) + len(their_passes), 1) * 100,
        "ppda": len(their_build_up) / max(len(ppda_actions), 1),
        "def_height": _x(my_def["location"]).mean() * YARD_TO_M,
        "recovery_height": _x(mine.loc[won, "location"]).mean() * YARD_TO_M,
        "long_ball_share": (own_half["pass_length"] >= 35).mean() * 100 if len(own_half) else np.nan,
        "pass_length": my_passes["pass_length"].mean() * YARD_TO_M,
        "crosses": my_passes["pass_cross"].eq(True).sum() * per90,
        "wide_entry_share": ((_y(entries) < 18) | (_y(entries) > 62)).mean() * 100 if len(entries) else np.nan,
        "through_balls": my_passes["pass_through_ball"].eq(True).sum() * per90,
        "fast_attacks": _fast_attacks(ev, team) * per90,
        "final_third_entries": len(entries) * per90,
        "box_entries_allowed": _box_entries(ev, opponent) * per90,
        "opp_shot_distance": opp_dist.mean() if len(opp_dist) else np.nan,
        # 결과: 동점 시간대의 90분당 xG (페널티 제외)
        "np_xg_for": my_shots["shot_statsbomb_xg"].sum() * per90,
        "np_xg_against": their_shots["shot_statsbomb_xg"].sum() * per90,
    }


def _finishing(ev: pd.DataFrame, team: str) -> dict:
    """경기 전체 기준 페널티 제외 득점과 xG. (결정력 = 득점 - xG)"""
    shots = ev[(ev["team"] == team) & (ev["type"] == "Shot") & (ev["shot_type"] != "Penalty")]
    return {"np_goals": shots["shot_outcome"].eq("Goal").sum(), "np_xg_full": shots["shot_statsbomb_xg"].sum()}


def team_match_table(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """경기 x 팀 한 줄씩: 동점 시간대 스타일 지표, 동점 시간대 xG, 경기 전체 결정력."""
    rows = []
    by_match = dict(tuple(events.groupby("match_id", sort=False)))  # 경기마다 한 번만 나눠 둔다
    for m in matches.itertuples():
        ev = by_match.get(m.match_id)
        if ev is None:
            continue
        ev = add_score_state(ev, m.home_team, m.away_team)
        level = ev[ev["score_diff"] == 0]
        level_minutes = level.groupby(["period", "minute"]).ngroups
        for team, opponent, venue in ((m.home_team, m.away_team, "홈"), (m.away_team, m.home_team, "원정")):
            row = {"match_id": m.match_id, "match_date": m.match_date, "team": team,
                   "opponent": opponent, "venue": venue, "level_minutes": level_minutes,
                   "goals_for": m.home_score if venue == "홈" else m.away_score,
                   "goals_against": m.away_score if venue == "홈" else m.home_score}
            if level_minutes >= MIN_LEVEL_MINUTES:
                row.update(_team_features(level, team, opponent, level_minutes))
            row.update(_finishing(ev, team))
            rows.append(row)
    table = pd.DataFrame(rows)

    # 상대 결정력을 같은 줄에 붙인다
    opp = table[["match_id", "team", "np_goals", "np_xg_full"]].rename(
        columns={"team": "opponent", "np_goals": "np_goals_against", "np_xg_full": "np_xg_full_against"})
    return table.merge(opp, on=["match_id", "opponent"])


def team_profiles(team_matches: pd.DataFrame) -> pd.DataFrame:
    """팀별 시즌 평균 스타일."""
    cols = list(STYLE_FEATURES) + ["np_xg_for", "np_xg_against", "goals_for", "goals_against"]
    return team_matches.groupby("team")[cols].mean()
