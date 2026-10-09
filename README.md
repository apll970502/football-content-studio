# Football Content Studio

LLM 기반 유튜브 콘텐츠 제작 자동화 프로젝트. 첫 적용 분야는 **축구 경기 분석 콘텐츠**입니다.

공개 축구 데이터(StatsBomb Open Data)로 선수·팀·전술을 수치로 분석하고,
그 결과를 근거로 LLM이 분석 보고서, 영상 대본, 제목·설명문을 만듭니다.
핵심 원칙은 **"숫자는 코드가, 글은 LLM이"** 입니다.

- 상세 개발 기획서: [docs/PROJECT_DEVELOPMENT_PLAN.md](docs/PROJECT_DEVELOPMENT_PLAN.md)
- 진행 상황: [PROGRESS.md](PROGRESS.md)

## 현재 프로토타입
| 파일 | 내용 |
|---|---|
| `app.py` | 선수 평가 화면 (Streamlit) |
| `data.py` | StatsBomb 데이터 수집·저장 |
| `metrics.py` | 선수 평가 (90분당 지표, 포지션 내 퍼센타일) |
| `style.py` | 팀 전술 스타일 지표 (동점 시간대 기준) |
| `tactics.py` | 스타일 분류, 전력 보정, 전술 상성 분석 |

## 실행
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```
처음 대회를 고르면 경기 데이터를 내려받아 `data/` 폴더에 저장합니다.

## 데이터 출처
Data provided by [StatsBomb](https://github.com/statsbomb/open-data) (Hudl StatsBomb Open Data).
