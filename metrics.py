"""이벤트 데이터로 선수 기록을 계산하는 분석 엔진. AI 없이 계산만으로 동작한다.

좌표 기준(StatsBomb): 경기장은 가로 120 x 세로 80이고,
모든 이벤트는 "그 행동을 한 팀이 오른쪽(x=120)으로 공격하는 방향"으로 기록되어 있다.
"""

import pandas as pd

# StatsBomb 세부 포지션 -> 비교용 포지션 그룹
POSITION_GROUPS = {
    "Goalkeeper": "골키퍼",
    "Center Back": "센터백", "Left Center Back": "센터백", "Right Center Back": "센터백",
    "Left Back": "풀백·윙백", "Right Back": "풀백·윙백",
    "Left Wing Back": "풀백·윙백", "Right Wing Back": "풀백·윙백",
    "Center Defensive Midfield": "중앙 미드필더", "Left Defensive Midfield": "중앙 미드필더",
    "Right Defensive Midfield": "중앙 미드필더", "Center Midfield": "중앙 미드필더",
    "Left Center Midfield": "중앙 미드필더", "Right Center Midfield": "중앙 미드필더",
    "Left Midfield": "윙어·공격형 미드필더", "Right Midfield": "윙어·공격형 미드필더",
    "Center Attacking Midfield": "윙어·공격형 미드필더",
    "Left Attacking Midfield": "윙어·공격형 미드필더",
    "Right Attacking Midfield": "윙어·공격형 미드필더",
    "Left Wing": "윙어·공격형 미드필더", "Right Wing": "윙어·공격형 미드필더",
    "Center Forward": "스트라이커", "Left Center Forward": "스트라이커",
    "Right Center Forward": "스트라이커", "Secondary Striker": "스트라이커",
}

# 계산하는 지표: 열 이름 -> 화면에 보일 이름. (모두 "높을수록 좋음")
METRICS = {
    "passes": "패스 시도",
    "pass_accuracy": "패스 성공률(%)",
    "progressive_passes": "전진 패스",
    "key_passes": "키패스(슈팅으로 이어진 패스)",
    "shots": "슈팅",
    "xg": "기대득점(xG)",
    "progressive_carries": "전진 드리블(볼 운반)",
    "dribbles_won": "드리블 돌파 성공",
    "box_touches": "페널티박스 안 터치",
    "pressures": "압박 시도",
    "ball_wins": "볼 탈취(태클·인터셉트·리커버리)",
}

# 이 비율 지표는 90분 환산을 하지 않는다.
RATE_METRICS = {"pass_accuracy"}

# 세트피스 패스는 전진 패스 계산에서 뺀다. (스로인·코너킥 등은 선수의 전진 능력과 관계가 적음)
SET_PIECE_PASSES = {"Throw-in", "Corner", "Free Kick", "Goal Kick", "Kick Off"}

RED_CARDS = {"Red Card", "Second Yellow"}


def _x(points: pd.Series) -> pd.Series:
    return points.str[0]


def _y(points: pd.Series) -> pd.Series:
    return points.str[1]


def _in_box(points: pd.Series) -> pd.Series:
    """상대 페널티박스 안(x 102~120, y 18~62)에 있는지."""
    x, y = _x(points), _y(points)
    return (x >= 102) & (y >= 18) & (y <= 62)


def minutes_played(events: pd.DataFrame) -> pd.DataFrame:
    """선수별 출전 시간(분)과 출전 경기 수.

    선발은 0분부터, 교체 투입은 투입 시점부터 시작하고,
    교체 아웃·퇴장 시점 또는 경기 종료 시점에 끝난다.
    """
    match_end = events.groupby("match_id")["minute"].max()

    # 출전 시작
    starts = []
    for _, row in events[events["type"] == "Starting XI"].iterrows():
        for slot in row["tactics"]["lineup"]:
            starts.append({"match_id": row["match_id"], "player_id": slot["player"]["id"], "start": 0})
    subs = events[events["type"] == "Substitution"]
    for _, row in subs.iterrows():
        starts.append({"match_id": row["match_id"], "player_id": row["substitution_replacement_id"],
                       "start": row["minute"]})
    spells = pd.DataFrame(starts)

    # 출전 종료 (교체 아웃, 퇴장)
    ends = subs[["match_id", "player_id", "minute"]]
    sent_off = pd.Series(False, index=events.index)
    for col in ("foul_committed_card", "bad_behaviour_card"):
        if col in events.columns:
            sent_off |= events[col].isin(RED_CARDS)
    cards = events[sent_off]
    ends = pd.concat([ends, cards[["match_id", "player_id", "minute"]]])
    ends = ends.groupby(["match_id", "player_id"])["minute"].min().rename("end").reset_index()

    spells = spells.merge(ends, on=["match_id", "player_id"], how="left")
    spells["end"] = spells["end"].fillna(spells["match_id"].map(match_end))
    spells["minutes"] = (spells["end"] - spells["start"]).clip(lower=0)

    return spells.groupby("player_id").agg(minutes=("minutes", "sum"), matches=("match_id", "nunique"))


def player_table(events: pd.DataFrame) -> pd.DataFrame:
    """선수 한 명당 한 줄로, 대회 전체 기록을 90분 기준으로 계산한 표."""
    ev = events[events["player_id"].notna()].copy()
    is_type = ev["type"].eq

    passes = is_type("Pass")
    pass_ok = passes & ev["pass_outcome"].isna()  # outcome이 비어 있으면 성공
    open_play_pass = passes & ~ev["pass_type"].isin(SET_PIECE_PASSES)
    carries = is_type("Carry")

    ev["passes"] = passes
    ev["passes_completed"] = pass_ok
    # 전진 패스: 성공한 오픈 플레이 패스 중 골대 방향으로 10m 이상 나아간 패스
    ev["progressive_passes"] = (open_play_pass & pass_ok
                                & (_x(ev["pass_end_location"]) - _x(ev["location"]) >= 10))
    ev["key_passes"] = passes & ev["pass_shot_assist"].eq(True)
    ev["shots"] = is_type("Shot")
    ev["xg"] = ev["shot_statsbomb_xg"].where(is_type("Shot"), 0).fillna(0)
    ev["progressive_carries"] = carries & (_x(ev["carry_end_location"]) - _x(ev["location"]) >= 10)
    ev["dribbles_won"] = is_type("Dribble") & ev["dribble_outcome"].eq("Complete")
    ev["box_touches"] = (is_type("Ball Receipt*") | carries | is_type("Shot")) & _in_box(ev["location"])
    ev["pressures"] = is_type("Pressure")
    ev["ball_wins"] = (
        (is_type("Duel") & ev["duel_type"].eq("Tackle")
         & ev["duel_outcome"].isin(["Won", "Success In Play", "Success Out"]))
        | (is_type("Interception") & ev["interception_outcome"].isin(["Won", "Success In Play", "Success Out"]))
        | is_type("Ball Recovery")
    )

    count_cols = [m for m in METRICS if m not in RATE_METRICS]
    totals = ev.groupby("player_id")[count_cols + ["passes_completed"]].sum()

    # 이름, 소속 팀, 가장 많이 뛴 포지션
    info = ev.groupby("player_id").agg(
        player=("player", "first"),
        team=("team", lambda s: s.mode().iloc[0]),
        position=("position", lambda s: s.mode().iloc[0] if s.notna().any() else None),
    )
    info["position_group"] = info["position"].map(POSITION_GROUPS)

    table = info.join(minutes_played(events)).join(totals)
    table = table[table["minutes"] > 0]

    table["pass_accuracy"] = table["passes_completed"] / table["passes"] * 100
    table = table.drop(columns="passes_completed")
    for col in count_cols:
        table[col] = table[col] / table["minutes"] * 90
    return table.reset_index()


def add_percentiles(table: pd.DataFrame, min_minutes: int) -> pd.DataFrame:
    """같은 포지션 그룹 안에서 각 지표가 상위 몇 %인지(0~100) 계산한다.

    출전 시간이 min_minutes 이상인 선수끼리만 비교한다.
    """
    pool = table[table["minutes"] >= min_minutes].copy()
    for col in METRICS:
        pool[f"{col}_pct"] = pool.groupby("position_group")[col].rank(pct=True) * 100
    return pool
