---

# 🏐 KOVO V-League AI Match Predictor

**Advanced Volleyball Match Prediction System with Meta-Drift Analysis & Auto-Tuning**

이 프로젝트는 한국 프로배구(KOVO) 데이터를 수집, 분석하여 경기 승패와 점수 차(Handicap)를 예측하는 머신러닝 시스템입니다. 단순히 승률만 계산하는 것이 아니라, **시즌별 메타 변화(Meta Drift)**를 감지하고 성별(남/녀)에 따라 **베팅 기준점(Threshold)을 자동 최적화**하여 실전 베팅 가이드를 제공합니다.

---

## 🔥 Key Features (핵심 기능)

### 1. 🧠 Dual Model Architecture (이중 예측 모델)

- **Classifier (승패 예측):** `Logistic Regression`을 사용하여 팀의 승리 확률을 계산합니다.
- **Regressor (득실차 예측):** `Ridge Regression (Positive Constraint)`을 사용하여 예상 점수 차를 계산합니다.
- **Logic:** 승률이 높더라도 득실차가 적으면 "접전(3:2)"으로 판단하여 리스크를 관리합니다.

### 2. 📈 Meta-Drift Analysis (시즌 메타 분석)

- 과거 10년 치 데이터를 맹신하지 않습니다. **최근 3시즌** 데이터만 선별하여 학습합니다.
- **Trend Monitoring:** "3년 평균 가중치"와 "이번 시즌 가중치"를 비교하여 **현재 메타**를 파악합니다.
- _예: 이번 시즌은 서브(Serve)보다 공격성공률(Attack)이 승패에 더 큰 영향을 미침._

- 예측 시, ELO 승자와 메타(공격력) 우위 팀이 다를 경우 **[메타 경고]**를 출력합니다.

### 3. 🎯 Gender-Specific Auto-Tuning (성별 맞춤형 자동 최적화)

- 남배(혼전 양상)와 여배(양극화 양상)의 특성이 다름을 인지합니다.
- **Grid Search**를 통해 각 성별에 맞는 최적의 **'승률 커트라인'**과 **'마핸 기준점'**을 스스로 찾아냅니다.
- _Auto-Logic:_ "남배는 승률 60%만 넘어도 진입 가능, 여배는 75%는 넘어야 안전" 등의 기준을 AI가 수립.

### 4. 🛡️ Smart Betting Guide (베팅 가이드)

- 단순 예측 결과를 넘어, 구체적인 행동 강령을 제시합니다.
- **💎 강력 추천:** 승률 & 득실차 & 메타가 모두 일치할 때 (마핸/일반승).
- **🍯 꿀통 (플핸):** 강팀이지만 최근 폼이 떨어져 "3:2 접전"이 예상될 때 역배/플핸 추천.
- **🚫 패스:** AI 확신도가 기준치 미만일 때 과감한 패스 권고.

### 5. ⚖️ Custom ELO System

- 세트 스코어와 관계없이 승패 기반의 ELO Rating을 자체 산출하여, "진짜 강팀(체급)"과 "반짝 강팀(거품)"을 구별합니다.

---

## 📂 Project Structure

파이프라인 순서대로 실행해야 합니다.

| 순서   | 파일명         | 설명                                              | 비고                               |
| ------ | -------------- | ------------------------------------------------- | ---------------------------------- |
| **01** | `schedule.py`  | KOVO 공식 홈페이지에서 경기 일정 및 결과 크롤링   | `kovo_schedule_result.csv` 생성    |
| **02** | `collect.py`   | 각 경기의 상세 기록(공격, 블로킹, 범실 등) 수집   | `kovo_player_stats_final.csv` 생성 |
| **03** | `process.py`   | 데이터 전처리, 변수명 표준화(`gdate`, `hname` 등) | `kovo_analysis_ready.csv` 생성     |
| **04** | `train.py`     | 모델 학습, 메타 분석, 최적 기준점(Threshold) 탐색 | `kovo_dual_model.pkl` 모델 저장    |
| **05** | `predict_t.py` | 저장된 모델을 불러와 오늘/내일 경기 예측          | **최종 실행 파일**                 |

---

## 🚀 How to Use (사용법)

### Step 1. 데이터 최신화 (경기 있는 날마다 실행)

먼저 최신 경기 결과를 업데이트하고, 분석용 데이터를 만듭니다.

```bash
python schedule.py
# (필요시 collect.py 실행하여 상세 스탯 수집)
python process.py

```

### Step 2. 모델 학습 & 메타 분석 (주기적으로 실행)

최신 데이터를 바탕으로 AI에게 현재 트렌드를 학습시킵니다.

```bash
python train.py

```

> **Output 예시:**
>
> - 📊 시즌 메타 분석 리포트: `공격력(Diff_Att) 중요도 급상승 🔥`
> - 🧠 성별 최적 기준점: `남배 승률 컷 0.65, 여배 승률 컷 0.70`

### Step 3. 승부 예측 (매일 실행)

오늘 예정된 경기를 분석합니다.

```bash
python predict_t.py

```

> **Output 예시:**
>
> - 🏐 [‍♂️] 대한항공 vs 현대캐피탈
> - 🏆 승자: 대한항공 (62.3%) | 기준점: 65%
> - 🔢 예상 스코어: 3:2 (혼전/역배 주의)
> - 💡 가이드: 👉 승패 패스 / 👉 현대캐피탈 +1.5 플핸 추천

---

## 📊 Data Dictionary (Standardized)

모든 데이터는 아래 변수명으로 통일되어 관리됩니다.

- `gdate`: 경기 날짜 (YYYY-MM-DD)
- `seasonCode`: 시즌 코드 (예: 022)
- `hname` / `aname`: 홈팀명 / 원정팀명
- `score`: 세트 스코어 (예: 3 : 1)
- `ats` / `att`: 공격 성공 / 공격 시도
- `bs`: 블로킹 성공
- `ss`: 서브 성공
- `err`: 범실
- `diff_elo`: 두 팀 간의 ELO Rating 차이

---

## ⚠️ Disclaimer

이 시스템은 과거 데이터를 기반으로 한 통계적 예측 도구입니다. 경기 결과는 당일 선수의 컨디션, 부상, 심판 판정 등 예측 불가능한 변수에 의해 달라질 수 있습니다. **베팅의 책임은 전적으로 사용자 본인에게 있습니다.**
