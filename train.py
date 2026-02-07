import pandas as pd
import numpy as np
import pickle
import os

DATA_FILE = "kovo_analysis_ready.csv"
MODEL_FILE_MALE = "elo_model_male.pkl"
MODEL_FILE_FEMALE = "elo_model_female.pkl"

def calculate_elo_score(df, w_30, w_32):
    teams = pd.concat([df['h_std'], df['a_std']]).unique()
    elo_dict = {t: 1500 for t in teams}
    base_k = 20
    history = []

    for _, row in df.iterrows():
        h, a = row['h_std'], row['a_std']
        elo_h, elo_a = elo_dict[h], elo_dict[a]
        try:
            s_h, s_a = map(int, row['score'].split(':'))
        except: continue
        w_h = 1 if s_h > s_a else 0
        diff_score = abs(s_h - s_a)
        
        # 가중치 적용
        if diff_score == 3: k = base_k * w_30
        elif diff_score == 2: k = base_k * 1.0
        else: k = base_k * w_32
        
        pred_win = 1 if elo_h > elo_a else 0
        is_fav_win = (pred_win == w_h)
        
        history.append({
            'gdate': row['gdate'],
            'diff': abs(elo_h - elo_a),
            'fav_won': 1 if is_fav_win else 0,
            'dom_won': 1 if (is_fav_win and diff_score >= 2) else 0,
            'win_type': ("Dominant" if diff_score >= 2 else "Close") if is_fav_win else "Loss"
        })

        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        elo_dict[h] += k * (w_h - exp_h)
        elo_dict[a] += k * ((1 - w_h) - (1 - exp_h))
        
    acc = sum(x['fav_won'] for x in history) / len(history) if history else 0
    return acc, elo_dict, pd.DataFrame(history)

def analyze_cumulative_results(gender, history_df):
    print(f"\n📊 [{gender.upper()} - 누적 확률 분석 (Cumulative)]")
    print("=" * 80)
    print(f"{'ELO Diff 이상':^15} | {'전체 경기':^10} | {'승률 (Win%)':^15} | {'마핸율 (Dom%)':^15}")
    print("=" * 80)
    
    # 0부터 200까지 20단위로 누적 승률 계산
    # "ELO 차이가 X점 이상일 때 승률이 얼마인가?"
    thresholds = range(0, 201, 20)
    
    for th in thresholds:
        subset = history_df[history_df['diff'] >= th]
        total = len(subset)
        if total < 5: continue # 표본 너무 적으면 패스
        
        wins = subset['fav_won'].sum()
        doms = subset['dom_won'].sum()
        
        win_rate = (wins / total * 100)
        dom_rate = (doms / total * 100)
        
        # 시각적 바
        bar = "█" * int(dom_rate // 10)
        
        print(f"{th:>10}+   | {total:^10} | {win_rate:>13.1f}% | {bar} {dom_rate:>5.1f}%")
    print("=" * 80)

def train_model():
    print("🚀 [성별 분리 최적화 및 누적 분석]...")
    df = pd.read_csv(DATA_FILE)
    
    # 남자부
    df_male = df[df['gender'] == 'Male'].copy()
    params_m = (1.3, 0.6)
    acc_m, elo_m, hist_m = calculate_elo_score(df_male, params_m[0], params_m[1])
    print(f"\n♂️ [남자부] 가중치 {params_m} | 적중률: {acc_m*100:.2f}%")
    analyze_cumulative_results("Male", hist_m)
    with open(MODEL_FILE_MALE, 'wb') as f: pickle.dump({'elo': elo_m, 'weights': params_m, 'last_date': df['gdate'].max()}, f)

    # 여자부
    df_female = df[df['gender'] == 'Female'].copy()
    params_f = (1.3, 0.7)
    acc_f, elo_f, hist_f = calculate_elo_score(df_female, params_f[0], params_f[1])
    print(f"\n♀️ [여자부] 가중치 {params_f} | 적중률: {acc_f*100:.2f}%")
    analyze_cumulative_results("Female", hist_f)
    with open(MODEL_FILE_FEMALE, 'wb') as f: pickle.dump({'elo': elo_f, 'weights': params_f, 'last_date': df['gdate'].max()}, f)
    
    print(f"\n💾 모델 분리 저장 완료")

if __name__ == "__main__":
    train_model()