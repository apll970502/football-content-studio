"""축구 분석기 - 선수 평가 화면.

실행: .venv\\Scripts\\streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

import data
import metrics

st.set_page_config(page_title="축구 분석기", page_icon=":material/sports_soccer:", layout="wide")


@st.cache_data(max_entries=5, show_spinner=False)
def load_player_table(competition_id: int, season_id: int) -> pd.DataFrame:
    events = data.load_competition_events(competition_id, season_id)
    return metrics.player_table(events)


# ---------- 사이드바: 대회와 선수 고르기 ----------
with st.sidebar:
    st.header(":material/sports_soccer: 축구 분석기")
    competition = st.selectbox("대회", list(data.COMPETITIONS))
    competition_id, season_id = data.COMPETITIONS[competition]

if not data.is_cached(competition_id, season_id):
    st.info("이 대회는 처음 불러오는 거라 경기 데이터를 내려받고 있어요. 한 번만 받으면 다음부터는 바로 열려요.",
            icon=":material/download:")
    bar = st.progress(0.0)
    data.load_competition_events(
        competition_id, season_id,
        on_progress=lambda done, total: bar.progress(done / total, text=f"{done} / {total} 경기"),
    )
    bar.empty()

with st.spinner("선수 기록을 계산하는 중이에요"):
    table = load_player_table(competition_id, season_id)

with st.sidebar:
    min_minutes = st.slider("비교 대상 최소 출전 시간(분)", 0, 900, 180, step=30,
                            help="출전 시간이 너무 짧은 선수는 기록이 들쭉날쭉해서 비교에서 뺍니다.")
    pool = metrics.add_percentiles(table, min_minutes)

    team = st.selectbox("팀", sorted(pool["team"].unique()))
    team_players = pool[pool["team"] == team].sort_values("minutes", ascending=False)
    if team_players.empty:
        st.warning("이 팀에는 최소 출전 시간을 넘긴 선수가 없어요. 출전 시간을 낮춰 보세요.")
        st.stop()
    player_id = st.selectbox(
        "선수", team_players["player_id"],
        format_func=lambda pid: (lambda r: f"{r['player']} · {r['position_group']} · {r['minutes']:.0f}분")(
            team_players.set_index("player_id").loc[pid]),
    )

# ---------- 본문: 선수 평가 ----------
player = pool.set_index("player_id").loc[player_id]
group = player["position_group"]
group_pool = pool[pool["position_group"] == group]

st.title(player["player"])
st.caption(f"{competition} · {player['team']} · {group} ({player['position']})")

with st.container(horizontal=True):
    st.metric("출전 시간", f"{player['minutes']:.0f}분", border=True)
    st.metric("출전 경기", f"{player['matches']:.0f}경기", border=True)
    st.metric("비교 대상", f"같은 포지션 {len(group_pool)}명", border=True)

# 지표별 퍼센타일 표 (차트와 강점·약점에 같이 씀)
rows = []
for col, label in metrics.METRICS.items():
    rows.append({
        "지표": label,
        "값": player[col],
        "포지션 평균": group_pool[col].mean(),
        # 포지션 안 순위로 계산해야 1위가 "상위 0%"가 아니라 "상위 3%"처럼 나온다.
        "상위 %": ((group_pool[col] > player[col]).sum() + 1) / len(group_pool) * 100,
        "퍼센타일": player[f"{col}_pct"],
    })
profile = pd.DataFrame(rows)[["지표", "값", "포지션 평균", "퍼센타일", "상위 %"]]

st.subheader("같은 포지션과 비교")
st.caption(f"막대가 길수록 같은 포지션({group}) 선수들보다 잘한 지표예요. 50이 평균이에요. "
           "패스 성공률을 뺀 나머지는 90분 기준 기록이에요.")
chart = (
    alt.Chart(profile)
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X("퍼센타일:Q", scale=alt.Scale(domain=[0, 100]), title="퍼센타일 (0~100)"),
        y=alt.Y("지표:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color("퍼센타일:Q", scale=alt.Scale(domain=[0, 50, 100], range=["#d1495b", "#d9d9d9", "#2a9d8f"]),
                        legend=None),
        tooltip=["지표", alt.Tooltip("값:Q", format=".2f"), alt.Tooltip("포지션 평균:Q", format=".2f"),
                 alt.Tooltip("퍼센타일:Q", format=".0f")],
    )
)
rule = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
st.altair_chart(chart + rule)

strong, weak = st.columns(2)
with strong.container(border=True):
    st.markdown("**:material/trending_up: 강점** (상위 25% 이내)")
    top = profile[profile["퍼센타일"] >= 75].sort_values("퍼센타일", ascending=False)
    for _, r in top.iterrows():
        st.markdown(f"- {r['지표']}: 상위 {r['상위 %']:.0f}% ({r['값']:.2f}, 평균 {r['포지션 평균']:.2f})")
    if top.empty:
        st.caption("상위 25%에 드는 지표가 없어요.")
with weak.container(border=True):
    st.markdown("**:material/trending_down: 약점** (하위 25% 이내)")
    bottom = profile[profile["퍼센타일"] <= 25].sort_values("퍼센타일")
    for _, r in bottom.iterrows():
        st.markdown(f"- {r['지표']}: 하위 {r['퍼센타일']:.0f}% ({r['값']:.2f}, 평균 {r['포지션 평균']:.2f})")
    if bottom.empty:
        st.caption("하위 25%에 드는 지표가 없어요.")

with st.expander("전체 기록 표"):
    st.dataframe(
        profile.drop(columns="상위 %"),
        hide_index=True,
        alt=f"{player['player']}의 지표별 기록, 포지션 평균, 퍼센타일",
        column_config={
            "값": st.column_config.NumberColumn(format="%.2f"),
            "포지션 평균": st.column_config.NumberColumn(format="%.2f"),
            "퍼센타일": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        },
    )
