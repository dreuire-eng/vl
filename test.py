import pandas as pd
import numpy as np
import pickle
import os

# =========================================================
# 1. 설정
# =========================================================
DATA_FILE = "kovo_analysis_ready.csv"
MODEL_FILE = "elo_model.pkl"

def analyze_victory_quality():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
        print("❌ 필수 파일이 없습니다.")
        return

    # 1. 모델 로드 (가중치 확인용)
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    
    # 가중치 가져오기 (없으면 기본값)
    if isinstance(model, dict):
        weights = model.get('weights', (1.2, 0.8))
    else:
        weights = (1.2, 0.8)

    w_30, w_32 = weights
    print(f"⚙️ 적용 가중치: 3:0({w_30}), 3:2({w_32})")

    # 2. 데이터 로드 및 ELO 재계산
    df = pd.read_csv(DATA_FILE)
    elo_dict = {t: 1500 for t in pd.concat([df['h_std'], df['a_std']]).unique()}
    history = []
    base_k = 20

    for _, row in df.iterrows():
        h, a = row['h_std'], row['a_std']
        elo_h, elo_a = elo_dict[h], elo_dict[a]
        
        try:
            s_h, s_a = map(int, row['score'].split(':'))
        except: continue
        
        diff_score = abs(s_h - s_a)
        w_h = 1 if s_h > s_a else 0
        
        # 정배/역배 판별
        if elo_h > elo_a:
            is_fav_win = (w_h == 1)
            elo_diff = elo_h - elo_a
        else:
            is_fav_win = (w_h == 0)
            elo_diff = elo_a - elo_h
            
        # 승리 형태 (정배가 이겼을 때만 기록)
        if is_fav_win:
            if diff_score >= 2: 
                win_type = "Dominant" # 3:0, 3:1
            else: 
                win_type = "Close"    # 3:2
            
            history.append({
                'elo_diff': elo_diff,
                'win_type': win_type
            })

        # ELO 업데이트
        if diff_score == 3: k = base_k * w_30
        elif diff_score == 2: k = base_k * 1.0
        else: k = base_k * w_32
        
        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        elo_dict[h] += k * (w_h - exp_h)
        elo_dict[a] += k * ((1 - w_h) - (1 - exp_h))

    # 3. 구간별 분석 (0~500, 25단위)
    df_res = pd.DataFrame(history)
    bins = list(range(0, 501, 25))
    labels = [f"{i}~{i+25}" for i in range(0, 500, 25)]
    
    df_res['bin'] = pd.cut(df_res['elo_diff'], bins=bins, labels=labels, right=False)
    
    # 집계
    stats = df_res.groupby('bin', observed=False)['win_type'].value_counts().unstack(fill_value=0)
    
    if 'Dominant' not in stats.columns: stats['Dominant'] = 0
    if 'Close' not in stats.columns: stats['Close'] = 0
    
    stats['Total_Wins'] = stats['Dominant'] + stats['Close']
    stats['Dom_Rate'] = (stats['Dominant'] / stats['Total_Wins'] * 100).fillna(0)
    stats['Close_Rate'] = (stats['Close'] / stats['Total_Wins'] * 100).fillna(0)

    # 4. 출력
    print("\n📊 [정배 승리 시 '우세 vs 접전' 비율 분석]")
    print(f"{'ELO Diff':^10} | {'Wins':^5} | {'우세 (3:0/3:1)':^14} | {'접전 (3:2)':^14} | {'우세 강도(Bar)':^20}")
    print("=" * 85)
    
    for idx, row in stats.iterrows():
        total = int(row['Total_Wins'])
        dom = int(row['Dominant'])
        close = int(row['Close'])
        dom_rate = row['Dom_Rate']
        close_rate = row['Close_Rate']
        
        # 그래프 바 (우세 비율 시각화)
        bar_len = int(dom_rate // 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        if total > 0:
            print(f"{idx:^10} | {total:>5} | {dom_rate:>5.1f}% ({dom:>3}) | {close_rate:>5.1f}% ({close:>3}) | {bar}")
        else:
            print(f"{idx:^10} | {total:>5} |       -        |       -        | {'-':^20}")

    print("=" * 85)
    print("📌 해석 가이드:")
    print(" - 우세(Dominant): 핸디캡 승리 (마핸) 가능성")
    print(" - 접전(Close): 일반승은 했지만 핸디캡은 패배 (프핸) 가능성")
    print(" 👉 우세 비율이 75~80% 이상으로 안정되는 구간을 '풀베팅' 기준으로 잡으세요.")

if __name__ == "__main__":
    analyze_victory_quality()