"""추적 데이터(모든 선수 위치)로 실제 공격 전개를 2D 애니메이션 영상(MP4)으로 만든다.

데이터: Metrica Sports 공개 샘플 (초당 25프레임, 좌표는 0~1로 정규화, y=0이 화면 위쪽)
실행:  .venv\\Scripts\\python animation\\play_animation.py
"""

from pathlib import Path

import imageio_ffmpeg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mplsoccer import Pitch

matplotlib.rcParams["font.family"] = "Malgun Gothic"  # 한글 표시 (Windows 기본 글꼴)
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "metrica"
OUTPUT = ROOT / "output"
PITCH_LENGTH, PITCH_WIDTH = 105, 68
FPS = 25
COLORS = {"Home": "#e63946", "Away": "#4ea8de"}


def load_tracking(game: int, team: str) -> pd.DataFrame:
    """한 팀의 추적 데이터를 미터 단위로 읽는다. 열: {team}_{등번호}_x / _y, ball_x / ball_y."""
    path = DATA / f"Sample_Game_{game}_RawTrackingData_{team}_Team.csv"
    jerseys = pd.read_csv(path, skiprows=1, nrows=1, header=None).iloc[0]
    raw = pd.read_csv(path, skiprows=2)

    columns = list(raw.columns[:3])  # Period, Frame, Time [s]
    for i in range(3, len(raw.columns) - 2, 2):
        number = int(jerseys[i])
        columns += [f"{team}_{number}_x", f"{team}_{number}_y"]
    columns += ["ball_x", "ball_y"]
    raw.columns = columns
    raw = raw.set_index("Frame")

    # 0~1 좌표를 미터로. y는 아래가 0이 되도록 뒤집는다 (그림에서 위쪽이 위로 보이게)
    for col in raw.columns:
        if col.endswith("_x"):
            raw[col] = raw[col] * PITCH_LENGTH
        elif col.endswith("_y"):
            raw[col] = (1 - raw[col]) * PITCH_WIDTH
    return raw


def load_events(game: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"Sample_Game_{game}_RawEventsData.csv")


def player_columns(df: pd.DataFrame, team: str) -> list[str]:
    return sorted({c.rsplit("_", 1)[0] for c in df.columns if c.startswith(team + "_")},
                  key=lambda p: int(p.split("_")[1]))


def describe(event: pd.Series) -> str:
    """이벤트 한 줄 설명 (화면 아래 자막)."""
    kind = {"PASS": "패스", "SHOT": "슈팅", "BALL LOST": "공 뺏김", "RECOVERY": "볼 탈취",
            "CHALLENGE": "경합", "SET PIECE": "세트피스", "BALL OUT": "아웃"}.get(event["Type"], event["Type"])
    who = str(event["From"]).replace("Player", "")
    to = f" → {str(event['To']).replace('Player', '')}번" if isinstance(event["To"], str) else ""
    extra = " · 골!" if isinstance(event["Subtype"], str) and "GOAL" in event["Subtype"] else ""
    team = "홈" if event["Team"] == "Home" else "원정"
    return f"{team} {who}번 {kind}{to}{extra}"


def render_play(game: int, end_frame: int, seconds_before: float, seconds_after: float,
                title: str, highlight: str, out_name: str) -> Path:
    home, away = load_tracking(game, "Home"), load_tracking(game, "Away")
    tracking = home.join(away.drop(columns=["Period", "Time [s]", "ball_x", "ball_y"]))
    events = load_events(game)

    start = end_frame - int(seconds_before * FPS)
    stop = end_frame + int(seconds_after * FPS)
    clip = tracking.loc[start:stop]
    window_events = events[(events["Start Frame"] >= start) & (events["Start Frame"] <= stop)]

    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                  pitch_color="#1b4332", line_color="#d8f3dc", linewidth=1.5)
    fig, ax = plt.subplots(figsize=(12, 8.6))
    pitch.draw(ax=ax)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.08)
    fig.set_facecolor("#0b2117")
    fig.suptitle(title, color="white", fontsize=17, fontweight="bold", y=0.96)
    caption = fig.text(0.5, 0.03, "", ha="center", color="white", fontsize=15)
    clock = ax.text(2, PITCH_WIDTH + 1.5, "", color="#d8f3dc", fontsize=10)

    dots, labels = {}, {}
    for team in ("Home", "Away"):
        for p in player_columns(clip, team):
            is_hl = p == highlight
            dots[p], = ax.plot([], [], "o", ms=13 if is_hl else 10, color=COLORS[team],
                               mec="#ffd60a" if is_hl else "white", mew=2.5 if is_hl else 1, zorder=3)
            labels[p] = ax.text(0, 0, p.split("_")[1], color="white", fontsize=7, ha="center",
                                va="center", fontweight="bold", zorder=4)
    ball, = ax.plot([], [], "o", ms=7, color="white", mec="black", zorder=5)
    trail, = ax.plot([], [], "-", lw=2, color="#ffd60a", alpha=0.7, zorder=2)

    frames = list(clip.index)

    def update(i):
        frame = frames[i]
        row = clip.loc[frame]
        for p, dot in dots.items():
            x, y = row[f"{p}_x"], row[f"{p}_y"]
            visible = not (np.isnan(x) or np.isnan(y))
            dot.set_data([x] if visible else [], [y] if visible else [])
            labels[p].set_visible(visible)  # 화면 밖 선수는 번호도 숨긴다
            if visible:
                labels[p].set_position((x, y))
        ball.set_data([row["ball_x"]], [row["ball_y"]])
        past = clip.loc[max(start, frame - FPS):frame]  # 최근 1초 공 궤적
        trail.set_data(past["ball_x"], past["ball_y"])
        recent = window_events[window_events["Start Frame"] <= frame]
        caption.set_text(describe(recent.iloc[-1]) if len(recent) else "")
        t = row["Time [s]"]
        clock.set_text(f"{int(t // 60):02d}:{t % 60:04.1f}")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS)
    OUTPUT.mkdir(exist_ok=True)
    out = OUTPUT / out_name
    anim.save(out, writer=FFMpegWriter(fps=FPS, bitrate=2500))
    plt.close(fig)
    return out


if __name__ == "__main__":
    events = load_events(2)
    goals = events[(events["Type"] == "SHOT") & events["Subtype"].str.contains("GOAL", na=False)]
    first = goals.iloc[0]
    shooter = f"{first['Team']}_{first['From'].replace('Player', '')}"
    path = render_play(
        game=2, end_frame=int(first["End Frame"]), seconds_before=14, seconds_after=2,
        title=f"Metrica 샘플 2경기 · 첫 골 장면 ({'홈' if first['Team'] == 'Home' else '원정'} "
              f"{first['From'].replace('Player', '')}번)",
        highlight=shooter, out_name="goal_demo.mp4",
    )
    print("저장:", path)
