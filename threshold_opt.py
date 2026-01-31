import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score, precision_score, recall_score

# =========================================================
# 1. 설정 및 데이터 로드
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")
MODEL_FILE = os.path.join(BASE_DIR, "kovo_dual_model.pkl")

def get_standardized_name(name):
    if pd.isna(name): return ""
    name_upper = str(name).upper().replace(" ", "")
    mapping = {
        '대한항공': ['KOREANAIR', 'JUMBOS', 'KAL', '대한항공'],
        '현대캐피탈': ['HYUNDAICAPITAL', 'SKYWALKERS', '현대캐피탈'],
        'KB손해보험': ['KBSTARS', 'KBINSURANCE', 'LIG', 'KB손해보험'],
        'OK금융그룹': ['OKFINANCIAL', 'OKSAVINGS', 'OKMAN', 'OK금융', '읏맨'],
        '한국전력': ['KEPCO', 'VIXTORM', 'KOREAELECTRIC', '한국전력'],
        '우리카드': ['WOORICARD', 'WOORIWON', '우리카드'],
        '삼성화재': ['SAMSUNG', 'BLUEFANGS', '삼성화재'],
        '흥국생명': ['HEUNGKUK', 'PINKSPIDERS', '흥국생명'],
        '현대건설': ['HYUNDAIE&C', 'HILLSTATE', '현대건설'],
        '정관장': ['JUNGKWANJANG', 'REDSPARKS', 'KGC', '정관장'],
        'IBK기업은행': ['IBK', 'ALTOS', '기업은행'],
        'GS칼텍스': ['GSCALTEX', 'KIXX', 'GS칼텍스'],
        '도로공사': ['HIPASS', 'EXPRESSWAY', '도로공사'],
        '페퍼저축은행': ['PEPPER', 'AIPEPPERS', '페퍼저축은행']
    }
    for std, keys in mapping.items():
        if any(k in name_upper for k in keys): return std
    return name

def analyze_thresholds():
    print("🚀 승률 구간별 스코어 분포 분석 & 최적 임계값 찾기")
    print("-" * 60)

    # 1. 모델 로드
    if not os.path.exists(MODEL_FILE):
        print("❌ 모델 파일이 없습니다.")
        return
    with open(MODEL_FILE, "rb") as f: pkg = pickle.load(f)
    clf = pkg['classifier']
    scaler = pkg['scaler']
    features = pkg['features']

    # 2. 데이터 로드 및 전처리 (학습때와 동일)
    df = pd.read_csv(DATA_FILE)
    if 'set_score' in df.columns: df.rename(columns={'set_score': 'score'}, inplace=True)
    if 'team_name' in df.columns: df.rename(columns={'team_name': 'tsname'}, inplace=True)
    
    df['tsname'] = df['tsname'].astype(str)
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'game_num'])

    for c in ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 팀별 집계
    team_grp = df.groupby(['game_date', 'game_num', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'home_team': 'first', 'score': 'first'
    }).reset_index()

    team_grp['attack_rate'] = team_grp.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_grp['receive_rate'] = team_grp.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    
    team_grp['home_team_std'] = team_grp['home_team'].apply(get_standardized_name)
    team_grp['is_home'] = team_grp['team_std'] == team_grp['home_team_std']
    
    # 롤링 평균 계산
    team_grp = team_grp.sort_values(['team_std', 'game_date'])
    metrics = ['attack_rate', 'bs', 'ss', 'err', 'receive_rate']
    for m in metrics:
        team_grp[f'roll_{m}'] = team_grp.groupby('team_std')[m].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # 매치업 생성
    elo = {t: 1500 for t in team_grp['team_std'].unique()}
    matches = []
    
    sorted_games = team_grp.sort_values(['game_date', 'game_num'])
    for _, grp in sorted_games.groupby(['game_date', 'game_num']):
        if len(grp) != 2: continue
        
        h_rows = grp[grp['is_home'] == True]
        a_rows = grp[grp['is_home'] == False]
        if h_rows.empty or a_rows.empty: continue
        
        h, a = h_rows.iloc[0], a_rows.iloc[0]
        th, ta = h['team_std'], a['team_std']
        
        # 실제 결과 분석
        try:
            s = list(map(int, str(h['score']).split(':')))
            real_score_diff = s[0] - s[1] # 3, 2, 1, -1, -2, -3
        except: continue
        
        match_data = {
            'diff_elo': elo[th] - elo[ta],
            'diff_att': h['roll_attack_rate'] - a['roll_attack_rate'],
            'diff_block': h['roll_bs'] - a['roll_bs'],
            'diff_serve': h['roll_ss'] - a['roll_ss'],
            'diff_recv': h['roll_receive_rate'] - a['roll_receive_rate'],
            'diff_fault': h['roll_err'] - a['roll_err'],
            'real_diff': real_score_diff # +3이면 홈 3:0 승, -3이면 원정 3:0 승
        }
        matches.append(match_data)
        
        # ELO Update
        w_h = 1 if real_score_diff > 0 else 0
        exp_h = 1 / (1 + 10 ** ((elo[ta] - elo[th]) / 400))
        elo[th] += 20 * (w_h - exp_h)
        elo[ta] += 20 * ((1 - w_h) - (1 - exp_h))

    # 분석용 데이터프레임
    df_m = pd.DataFrame(matches).dropna()
    df_m['diff_fault'] = -df_m['diff_fault'] # 반전
    
    X = df_m[features]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    
    # 승리 확률 예측
    probs = clf.predict_proba(X_scaled)[:, 1]
    df_m['prob_home'] = probs
    
    # =========================================================
    # 📊 1. [Grid Search] 승률 구간별 실제 스코어 비율
    # =========================================================
    print("\n📊 1. 승률 구간별 스코어 출현 빈도 (Grid Search)")
    print(f"{'Prob Range':<15} | {'Games':<5} | {'3:0(%)':<8} | {'3:1(%)':<8} | {'3:2(%)':<8} | {'Upset(%)'}")
    print("-" * 75)
    
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.00]
    labels = ["50~55%", "55~60%", "60~65%", "65~70%", "70~75%", "75%~"]
    
    # 홈 승리 예측인 경기만 필터링 (승률 > 0.5)
    home_wins_pred = df_m[df_m['prob_home'] > 0.5].copy()
    home_wins_pred['prob_bin'] = pd.cut(home_wins_pred['prob_home'], bins=bins, labels=labels)
    
    for label in labels:
        subset = home_wins_pred[home_wins_pred['prob_bin'] == label]
        total = len(subset)
        if total == 0: continue
        
        cnt_30 = len(subset[subset['real_diff'] == 3])
        cnt_31 = len(subset[subset['real_diff'] == 2])
        cnt_32 = len(subset[subset['real_diff'] == 1])
        cnt_loss = len(subset[subset['real_diff'] < 0]) # 역배 터짐
        
        print(f"{label:<15} | {total:<5} | {cnt_30/total*100:>6.1f}%  | {cnt_31/total*100:>6.1f}%  | {cnt_32/total*100:>6.1f}%  | {cnt_loss/total*100:>6.1f}%")

    # =========================================================
    # 🎯 2. [Optimizer] F1-Score 기반 최적 임계값 찾기
    # =========================================================
    print("\n\n🎯 2. 최적 임계값 탐색 (F1-Score Maximization)")
    
    # (1) 3:0 셧아웃 기준선 찾기
    # 타겟: 실제로 3:0인 경기 (True) vs 나머지 (False)
    y_true_30 = (home_wins_pred['real_diff'] == 3).astype(int)
    probs_win = home_wins_pred['prob_home']
    
    best_th_30 = 0.5
    best_f1_30 = 0
    
    # 0.50부터 0.90까지 0.01 단위로 스캔
    for th in np.arange(0.50, 0.90, 0.01):
        y_pred = (probs_win >= th).astype(int)
        score = f1_score(y_true_30, y_pred, zero_division=0)
        if score > best_f1_30:
            best_f1_30 = score
            best_th_30 = th
            
    print(f"   🏆 [3:0 셧아웃] 최적 확률 기준: {best_th_30*100:.1f}% 이상")
    print(f"      (이 기준일 때 F1-Score가 {best_f1_30:.3f}로 최대)")

    # (2) 3:2 접전 기준선 찾기
    # 타겟: 실제로 3:2인 경기 (True) vs 나머지
    # 주의: 승률이 '낮을수록' 3:2 확률이 높으므로, "이 확률 이하면 3:2다"를 찾아야 함
    y_true_32 = (home_wins_pred['real_diff'] == 1).astype(int)
    
    best_th_32 = 0.6
    best_f1_32 = 0
    
    for th in np.arange(0.50, 0.70, 0.01):
        y_pred = (probs_win <= th).astype(int) # 확률이 th보다 '작으면' 3:2 예측
        score = f1_score(y_true_32, y_pred, zero_division=0)
        if score > best_f1_32:
            best_f1_32 = score
            best_th_32 = th
            
    print(f"   🏆 [3:2 풀세트] 최적 확률 기준: {best_th_32*100:.1f}% 미만")
    print(f"      (이 기준일 때 F1-Score가 {best_f1_32:.3f}로 최대)")
    
    print("\n   💡 [결론: 추천 가이드라인]")
    print(f"      - 승률 {best_th_32*100:.0f}% 미만 : 3:2 접전 (오버/핸디캡 추천)")
    print(f"      - 승률 {best_th_32*100:.0f}% ~ {best_th_30*100:.0f}% : 3:1 우세 (일반승 추천)")
    print(f"      - 승률 {best_th_30*100:.0f}% 이상 : 3:0 압승 (마핸/언더 추천)")

if __name__ == "__main__":
    analyze_thresholds()