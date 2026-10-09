"""전술 상성(파훼법) 분석 엔진. AI 없이 통계 계산만으로 동작한다.

흐름
1. 스타일 분류: 경기마다 팀이 쓴 전술(동점 시간대 스타일 지표)을 비슷한 것끼리 묶어 "스타일 유형"을 만든다.
2. 전력 보정: 팀마다 공격력·수비력을 추정해서, 그 경기에서 "원래 기대되는 xG 차이"를 계산한다.
3. 상성표: 실제 xG 차이 - 기대 xG 차이 = 전술이 만든 차이. 이것을 (상대 평소 스타일 x 우리가 쓴 스타일)별로 평균낸다.
   → 전력이 비슷하지 않아도 "어떤 스타일이 어떤 스타일에 강한가"를 비교할 수 있다.
4. 결정력: 득점 - xG. 전술이 아니라 선수 개인 능력(마무리, 선방)과 운이 만든 차이라서 따로 본다.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

# 스타일 분류에 쓰는 지표 (서로 겹치는 지표는 뺐다)
CLUSTER_FEATURES = [
    "possession", "ppda", "def_height", "recovery_height", "long_ball_share",
    "crosses", "wide_entry_share", "through_balls", "fast_attacks",
    "box_entries_allowed", "opp_shot_distance",
]

# 스타일 이름 짓기: 지표가 평균보다 확실히 높거나(+1) 낮을 때(-1) 붙는 말
DESCRIPTORS = {
    ("possession", +1): "점유형", ("possession", -1): "점유 포기",
    ("ppda", -1): "강한 압박", ("ppda", +1): "소극적 압박",
    ("def_height", +1): "높은 수비 라인", ("def_height", -1): "내려앉은 수비",
    ("recovery_height", +1): "전방 탈취", ("recovery_height", -1): "후방 탈취",
    ("long_ball_share", +1): "롱볼", ("long_ball_share", -1): "짧은 빌드업",
    ("crosses", +1): "크로스 위주", ("crosses", -1): "크로스 적음",
    ("wide_entry_share", +1): "측면 공략", ("wide_entry_share", -1): "중앙 공략",
    ("through_balls", +1): "뒷공간 침투",
    ("fast_attacks", +1): "빠른 전환",
    ("box_entries_allowed", -1): "박스 봉쇄", ("box_entries_allowed", +1): "박스 허용",
    ("opp_shot_distance", +1): "슈팅 거리 밀어내기",
}

MIN_RELIABLE = 30  # 이보다 경기 수가 적은 칸은 "신뢰 낮음"


def usable(team_matches: pd.DataFrame) -> pd.DataFrame:
    """동점 시간이 충분해서 스타일 지표가 계산된 경기만.

    비율 지표는 계산 대상이 없으면 비어 있다(예: 상대가 슈팅을 한 번도 못 함). 이런 칸은 리그 중앙값으로 채우고,
    상대 슈팅이 없었던 경우는 "좋은 위치를 전혀 안 내줬다"는 뜻이라 슈팅 거리를 리그 상위 25% 값으로 채운다.
    """
    df = team_matches[team_matches["possession"].notna()].copy()
    by_league = df.groupby("league")
    df["opp_shot_distance"] = df["opp_shot_distance"].fillna(by_league["opp_shot_distance"].transform("quantile", 0.75))
    for col in CLUSTER_FEATURES:
        df[col] = df[col].fillna(by_league[col].transform("median"))
    return df


def standardize(team_matches: pd.DataFrame) -> pd.DataFrame:
    """리그마다 평균 0, 표준편차 1로 맞춘다. (리그별 성향 차이를 지우고 팀 간 차이만 남긴다)"""
    return team_matches.groupby("league")[CLUSTER_FEATURES].transform(lambda s: (s - s.mean()) / s.std())


def name_style(center: pd.Series) -> str:
    """스타일 유형 중심값(z점수)으로 이름을 붙인다. 가장 두드러진 특징 2개."""
    picks = []
    for feat, z in center.abs().sort_values(ascending=False).items():
        word = DESCRIPTORS.get((feat, int(np.sign(center[feat]))))
        if word and z >= 0.4:
            picks.append(word)
        if len(picks) == 2:
            break
    return " · ".join(picks) if picks else "균형형"


def fit_styles(team_matches: pd.DataFrame, n_styles: int = 6, seed: int = 0):
    """경기별 스타일을 n_styles개 유형으로 묶는다. (유형 번호, 유형 이름표, 중심값, 군집 모델)."""
    z = standardize(team_matches)
    model = KMeans(n_clusters=n_styles, n_init=20, random_state=seed).fit(z)
    centers = pd.DataFrame(model.cluster_centers_, columns=CLUSTER_FEATURES)
    names = {i: name_style(centers.loc[i]) for i in centers.index}
    counts = {}
    for i, n in list(names.items()):  # 이름이 겹치면 번호를 붙여 구분
        if list(names.values()).count(n) > 1:
            counts[n] = counts.get(n, 0) + 1
            names[i] = f"{n} {counts[n]}"
    return pd.Series(model.labels_, index=team_matches.index), names, centers, model


def season_profiles(team_matches: pd.DataFrame) -> pd.DataFrame:
    """팀별 시즌 평균 스타일 (리그 안에서 표준화한 z점수)."""
    raw = team_matches.groupby(["league", "team"])[CLUSTER_FEATURES].mean()
    return raw.groupby(level="league").transform(lambda s: (s - s.mean()) / s.std())


def fit_season_styles(team_matches: pd.DataFrame, n_styles: int = 5, seed: int = 0):
    """상대를 분석할 때 쓰는 '평소 스타일' 유형. 팀 시즌 평균끼리 따로 묶는다.

    경기 하나하나는 들쭉날쭉해서, 시즌 평균을 경기 단위 유형에 맞추면 특징이 흐려진다.
    (팀별 유형 번호, 유형 이름표, 팀별 z점수)를 돌려준다.
    """
    profiles = season_profiles(team_matches)
    model = KMeans(n_clusters=n_styles, n_init=50, random_state=seed).fit(profiles)
    centers = pd.DataFrame(model.cluster_centers_, columns=CLUSTER_FEATURES)
    names = {i: name_style(centers.loc[i]) for i in centers.index}
    counts = {}
    for i, n in list(names.items()):
        if list(names.values()).count(n) > 1:
            counts[n] = counts.get(n, 0) + 1
            names[i] = f"{n} {counts[n]}"
    return pd.Series(model.labels_, index=profiles.index), names, profiles


def expected_xg_diff(team_matches: pd.DataFrame, alpha: float = 1.0) -> pd.Series:
    """전력만 보고 기대되는 xG 차이(우리 xG - 상대 xG, 90분 기준).

    리그마다 'xG = 우리 공격력 + 상대 수비 허용도 + 홈 이점'을 학습해서,
    두 팀 전력으로 예상되는 xG를 양쪽 모두 계산한 뒤 뺀다.
    """
    expected = pd.Series(index=team_matches.index, dtype=float)
    for _, df in team_matches.groupby("league"):
        X = pd.concat([
            pd.get_dummies(df["team"], prefix="atk"),
            pd.get_dummies(df["opponent"], prefix="def"),
            (df["venue"] == "홈").astype(int).rename("home"),
        ], axis=1).astype(float)
        # 동점 시간이 긴 경기일수록 xG가 믿을 만하므로 가중치를 더 준다
        model = Ridge(alpha=alpha).fit(X, df["np_xg_for"], sample_weight=df["level_minutes"])
        pred = pd.Series(model.predict(X), index=df.index)

        # 같은 경기에서 상대 입장의 예상 xG를 찾아 뺀다
        key = df["match_id"].astype(str) + "|" + df["team"]
        opp_key = df["match_id"].astype(str) + "|" + df["opponent"]
        by_key = pd.Series(pred.values, index=key.values)
        expected.loc[df.index] = pred.values - by_key.reindex(opp_key.values).values
    return expected


def add_effects(team_matches: pd.DataFrame) -> pd.DataFrame:
    """경기마다 전술 효과와 결정력을 붙인다."""
    actual = team_matches["np_xg_for"] - team_matches["np_xg_against"]
    return team_matches.assign(
        tactic_effect=actual - expected_xg_diff(team_matches),
        # 결정력: (우리 득점 - 우리 xG) - (상대 득점 - 상대 xG). 경기 전체 기준
        finishing_effect=(team_matches["np_goals"] - team_matches["np_xg_full"])
        - (team_matches["np_goals_against"] - team_matches["np_xg_full_against"]),
    )


def matchup_table(df: pd.DataFrame) -> pd.DataFrame:
    """(상대 평소 스타일 x 우리가 쓴 스타일)별 평균 전술 효과와 결정력.

    df에는 my_style, opp_style, tactic_effect, finishing_effect, level_minutes 열이 있어야 한다.
    전술 효과는 동점 시간이 긴 경기에 더 큰 가중치를 준 평균이다.
    """
    def summarize(g: pd.DataFrame) -> pd.Series:
        w = g["level_minutes"]
        mean = np.average(g["tactic_effect"], weights=w)
        sd = np.sqrt(np.average((g["tactic_effect"] - mean) ** 2, weights=w))
        return pd.Series({
            "tactic_effect": mean,
            "se": sd / np.sqrt(len(g)),  # 표본이 적을수록 커지는 오차 범위
            "finishing_effect": g["finishing_effect"].mean(),
            "count": len(g),
        })

    table = df.groupby(["opp_style", "my_style"]).apply(summarize, include_groups=False).reset_index()
    table["reliable"] = table["count"] >= MIN_RELIABLE
    return table
