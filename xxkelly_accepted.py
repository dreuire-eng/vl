import pandas as pd
import joblib
from datetime import datetime
import sys

# 설정
FILE_CLF = "kovo_model_clf.pkl"
FILE_REG = "kovo_model_reg.pkl"
DATA_FILE = "kovo_prediction_final_v4.csv"
SCHEDULE_FILE = "kovo_schedule_result.csv"

def calculate_kelly(win_prob, odds):
    if odds <= 1.0: return 0.0
    b = odds - 1
    p = win_prob
    q = 1 - p
    f = (b * p - q) / b
    if f < 0: return 0.0
    return f * 0.5 

def run_total_analysis():
    # 1. 로드
    try:
        clf = joblib.load(FILE_CLF)
        reg = joblib.load(FILE_REG)
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        schedule = pd.read_csv(SCHEDULE_FILE)
    except Exception as e:
        print(f"❌ 오류: {e}\n먼저 4단계(통합 학습)를 실행해주세요.")
        return

    # 2. 날짜 입력
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n📅 날짜 입력 (Enter = {today})")
    target_str = input(">> ").strip() or today
    target_date = pd.to_datetime(target_str)
    
    matches = schedule[schedule['gdate'] == target_str]
    if matches.empty:
        print("❌ 경기 없음")
        return

    print(f"\n🔥 {target_str} 통합 분석 리포트 (안전장치 적용됨)\n")

    for idx, row in matches.iterrows():
        h_team, a_team = row['hname'], row['aname']
        
        # --- 데이터 준비 ---
        h_hist = df[(df['home_team'] == h_team) | (df['away_team'] == h_team)]
        a_hist = df[(df['home_team'] == a_team) | (df['away_team'] == a_team)]
        h_hist = h_hist[h_hist['date'] < target_date]
        a_hist = a_hist[a_hist['date'] < target_date]

        if h_hist.empty or a_hist.empty:
            print(f"❌ 데이터 부족: {h_team} vs {a_team}")
            continue

        last_h = h_hist.sort_values('date').iloc[-1]
        last_a = a_hist.sort_values('date').iloc[-1]

        # 변수 계산
        metrics = ['att', 'recv', 'blk', 'srv', 'err']
        input_data = {}
        for m in metrics:
            val_h = last_h[f'home_avg_{m}'] if last_h['home_team'] == h_team else last_h[f'away_avg_{m}']
            val_a = last_a[f'home_avg_{m}'] if last_a['home_team'] == a_team else last_a[f'away_avg_{m}']
            input_data[f'diff_{m}'] = [val_h - val_a]

        rest_h = (target_date - last_h['date']).days - 1
        rest_a = (target_date - last_a['date']).days - 1
        input_data['diff_rest'] = [rest_h - rest_a]

        past_h2h = df[((df['home_team']==h_team) & (df['away_team']==a_team)) | ((df['home_team']==a_team) & (df['away_team']==h_team))]
        past_h2h = past_h2h[past_h2h['date'] < target_date]
        wins = sum(1 for _, r in past_h2h.iterrows() if (r['home_team']==h_team and r['score_diff']>0) or (r['away_team']==h_team and r['score_diff']<0))
        rate = wins / len(past_h2h) if not past_h2h.empty else 0.5
        input_data['h2h_win_rate_home'] = [rate]
        
        last_h_lp = last_h['home_lineup_power'] if last_h['home_team'] == h_team else last_h['away_lineup_power']
        last_a_lp = last_a['home_lineup_power'] if last_a['home_team'] == a_team else last_a['away_lineup_power']
        input_data['diff_lineup'] = [last_h_lp - last_a_lp]

        # --- 예측 실행 ---
        features = [f'diff_{m}' for m in metrics] + ['diff_rest', 'diff_lineup', 'h2h_win_rate_home']
        X_pred = pd.DataFrame(input_data)[features]

        win_prob = clf.predict_proba(X_pred)[0][1] # 홈 승률
        score_diff = reg.predict(X_pred)[0] # 점수차

        # --- 리포트 출력 ---
        print("="*60)
        print(f"🏐 {h_team} vs {a_team}")
        print("-" * 60)
        
        # 상세 지표 출력
        print(f"📊 주요 지표 우세 현황")
        print(f" - 🛌 휴식일: {'🏠 우위' if rest_h > rest_a else '✈️ 우위'} ({rest_h}일 vs {rest_a}일)")
        print(f" - ⚔️ 상대전적: {rate*100:.0f}% (홈 기준)")
        print(f" - 💪 라인업폼: {'🏠 우위' if last_h_lp > last_a_lp else '✈️ 우위'} (파워차이 {last_h_lp - last_a_lp:+.1f})")
        
        # 지표 표
        metric_names = {'att':'공격', 'recv':'리시브', 'blk':'블로킹', 'srv':'서브', 'err':'범실'}
        print(f"\n{'지표':<6} | {'홈':^6} vs {'원정':^6} | {'우세'}")
        print("-" * 40)
        for m in metrics:
            val_h = last_h[f'home_avg_{m}'] if last_h['home_team'] == h_team else last_h[f'away_avg_{m}']
            val_a = last_a[f'home_avg_{m}'] if last_a['home_team'] == a_team else last_a[f'away_avg_{m}']
            if m == 'err': marker = "🏠" if val_h < val_a else "✈️"
            else: marker = "🏠" if val_h > val_a else "✈️"
            print(f"{metric_names[m]:<6} | {val_h:6.2f} vs {val_a:6.2f} | {marker}")
        print("-" * 60)

        # 최종 결론
        print(f"🤖 [AI 최종 판단]")
        
        if win_prob > 0.5:
            prob_txt = f"🏠 홈팀 승리 유력 ({win_prob*100:.1f}%)"
            winner = "HOME"
            final_prob = win_prob
        else:
            prob_txt = f"✈️ 원정팀 승리 유력 ({(1-win_prob)*100:.1f}%)"
            winner = "AWAY"
            final_prob = 1 - win_prob
            
        print(f" 1️⃣ 일반 승패 : {prob_txt}")
        print(f" 2️⃣ 예상 스코어: {score_diff:+.2f} 세트 차이")
        
        # --- 🎯 [수정된] 베팅 전략 추천 ---
        print("\n🎯 [베팅 전략 추천]")
        
        # 1. 안전장치: 모델 간 의견 충돌 확인
        # (승률은 홈인데 점수는 마이너스거나, 승률은 원정인데 점수는 플러스인 경우)
        conflict = False
        if (winner == "HOME" and score_diff < 0) or (winner == "AWAY" and score_diff > 0):
            conflict = True
            
        if conflict:
            print(" ⚠️ [경고] 모델 의견 불일치! (승패 예측과 점수차 예측이 반대)")
            print(" 👉 판단 보류 (PASS 권장) ✋")
        else:
            # 의견이 일치할 때만 추천 로직 가동
            confidence = final_prob
            if confidence >= 0.65 and abs(score_diff) >= 1.0:
                print(f" 👉 [일반승] {winner} 승리! (강력 추천 ⭐⭐⭐)")
            elif confidence >= 0.55:
                print(f" 👉 [일반승] {winner} 승리 예상 (일반 추천 ⭐)")
            else:
                print(f" 👉 [일반승] 승패 난해함 (접전 예상 / 소액 추천)")

            # 핸디캡 힌트
            if abs(score_diff) >= 1.8:
                print(f" 👉 [핸디캡] {winner} 마핸(-1.5) 승리 가능성 높음!")
            elif abs(score_diff) <= 0.8:
                print(f" 👉 [핸디캡] {winner} 마핸 위험! (플핸 or 오버 추천)")

        print("="*60 + "\n")

if __name__ == "__main__":
    run_total_analysis()