import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score

# =========================================================
# 1. 설정 및 데이터 로드
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")
MODEL_FILE = os.path.join(BASE_DIR, "kovo_dual_model.pkl")

def get_standardized_name(name):
    if pd.isna(name): return ""
    name_str = str(name).upper().replace(" ", "")
    mapping = {
        '대한항공': ['KOREANAIR', 'JUMBOS', 'KAL', '대한항공', '점보스'],
        '현대캐피탈': ['HYUNDAICAPITAL', 'SKYWALKERS', '현대캐피탈', '스카이워커스'],
        'KB손해보험': ['KBSTARS', 'KBINSURANCE', 'LIG', 'KB손해보험', '케이비'],
        'OK금융그룹': ['OKFINANCIAL', 'OKSAVINGS', 'OKMAN', 'OK금융', '읏맨', 'OK'],
        '한국전력': ['KEPCO', 'VIXTORM', 'KOREAELECTRIC', '한국전력', '빅스톰'],
        '우리카드': ['WOORICARD', 'WOORIWON', '우리카드', '위비', 'WON'],
        '삼성화재': ['SAMSUNG', 'BLUEFANGS', '삼성화재', '블루팡스'],
        '흥국생명': ['HEUNGKUK', 'PINKSPIDERS', '흥국생명', '핑크스파이더스'],
        '현대건설': ['HYUNDAIE&C', 'HILLSTATE', '현대건설', '힐스테이트'],
        '정관장': ['JUNGKWANJANG', 'REDSPARKS', 'KGC', 'GINSENG', '정관장', '인삼공사'],
        'IBK기업은행': ['IBK', 'ALTOS', 'INDUSTRIALBANK', '기업은행', '알토스'],
        'GS칼텍스': ['GSCALTEX', 'KIXX', 'GS칼텍스', '킥스'],
        '도로공사': ['HIPASS', 'EXPRESSWAY', '도로공사', '하이패스'],
        '페퍼저축은행': ['PEPPER', 'AIPEPPERS', '페퍼저축은행', '페퍼']
    }
    for std, keys in mapping.items():
        if any(k in name_str for k in keys): return std
    return name_str

def analyze_point_thresholds():
    print("🚀 예상 득실차(Predicted Point Diff) 기반 스코어 분석")
    print("-" * 60)

    # 1. 모델 로드
    if not os.path.exists(MODEL_FILE):
        print("❌ 모델 파일이 없습니다.")
        return
    with open(MODEL_FILE, "rb") as f: pkg = pickle.load(f)
    clf = pkg['classifier']
    reg = pkg['regressor'] 
    scaler = pkg['scaler']
    features = pkg['features']

    # 2. 데이터 준비
    df = pd.read_csv(DATA_FILE)
    if 'set_score' in df.columns: df.rename(columns={'set_score': 'score'}, inplace=True)
    if 'team_name' in df.columns: df.rename(columns={'team_name': 'tsname'}, inplace=True)
    df['tsname'] = df['tsname'].astype(str)
    
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df = df.sort_values(['game_date', 'game_num'])

    for c in ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt', 'point']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    team_grp = df.groupby(['game_date', 'game_num', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'home_team': 'first', 'score': 'first',
        'point': 'sum'
    }).reset_index()

    team_grp['attack_rate'] = team_grp.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_grp['receive_rate'] = team_grp.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    team_grp['home_team_std'] = team_grp['home_team'].astype(str).apply(get_standardized_name)
    
    # 홈팀 여부 재확인
    team_grp['is_home'] = team_grp['team_std'] == team_grp['home_team_std']

    # Rolling Mean
    team_grp = team_grp.sort_values(['team_std', 'game_date'])
    metrics = ['attack_rate', 'bs', 'ss', 'err', 'receive_rate']
    for m in metrics:
        team_grp[f'roll_{m}'] = team_grp.groupby('team_std')[m].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    elo = {t: 1500 for t in team_grp['team_std'].unique()}
    matches = []

    sorted_games = team_grp.sort_values(['game_date', 'game_num'])
    
    for _, grp in sorted_games.groupby(['game_date', 'game_num']):
        if len(grp) != 2: continue
        
        # [수정] 인덱스 에러 방지용 안전장치
        h_rows = grp[grp['is_home'] == True]
        a_rows = grp[grp['is_home'] == False]
        
        if h_rows.empty or a_rows.empty:
            continue # 데이터 불량이면 패스
            
        h, a = h_rows.iloc[0], a_rows.iloc[0]
        th, ta = h['team_std'], a['team_std']
        
        try:
            s = list(map(int, str(h['score']).split(':')))
            real_set_diff = s[0] - s[1] 
        except: continue

        matches.append({
            'diff_elo': elo[th] - elo[ta],
            'diff_att': h['roll_attack_rate'] - a['roll_attack_rate'],
            'diff_block': h['roll_bs'] - a['roll_bs'],
            'diff_serve': h['roll_ss'] - a['roll_ss'],
            'diff_recv': h['roll_receive_rate'] - a['roll_receive_rate'],
            'diff_fault': h['roll_err'] - a['roll_err'],
            'real_set_diff': real_set_diff
        })
        
        w_h = 1 if real_set_diff > 0 else 0
        exp_h = 1 / (1 + 10 ** ((elo[ta] - elo[th]) / 400))
        elo[th] += 20 * (w_h - exp_h)
        elo[ta] += 20 * ((1 - w_h) - (1 - exp_h))

    # 분석 시작
    if not matches:
        print("❌ 분석할 경기 데이터가 없습니다.")
        return

    df_m = pd.DataFrame(matches).dropna()
    df_m['diff_fault'] = -df_m['diff_fault']
    
    X = df_m[features]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    
    # 예상 점수차 계산
    df_m['pred_point_diff'] = reg.predict(X_scaled)
    
    # 홈팀 승리 예상 경기만 필터링 (점수차 > 0)
    home_wins = df_m[df_m['pred_point_diff'] > 0].copy()
    
    print("\n📊 1. 예상 득실차 구간별 실제 결과 (Grid Search)")
    print(f"{'Pred Points':<15} | {'Games':<5} | {'3:0(%)':<8} | {'3:1(%)':<8} | {'3:2(%)':<8} | {'Fail(%)'}")
    print("-" * 75)
    
    bins = [0, 5, 10, 15, 20, 100]
    labels = ["0~5pts", "5~10pts", "10~15pts", "15~20pts", "20pts+"]
    
    home_wins['bin'] = pd.cut(home_wins['pred_point_diff'], bins=bins, labels=labels)
    
    for label in labels:
        subset = home_wins[home_wins['bin'] == label]
        total = len(subset)
        if total == 0: continue
        
        cnt_30 = len(subset[subset['real_set_diff'] == 3])
        cnt_31 = len(subset[subset['real_set_diff'] == 2])
        cnt_32 = len(subset[subset['real_set_diff'] == 1])
        cnt_fail = len(subset[subset['real_set_diff'] < 0])
        
        print(f"{label:<15} | {total:<5} | {cnt_30/total*100:>6.1f}%  | {cnt_31/total*100:>6.1f}%  | {cnt_32/total*100:>6.1f}%  | {cnt_fail/total*100:>6.1f}%")

    print("\n\n🎯 2. 최적 점수 기준 탐색")
    
    # (1) 3:0 셧아웃 (마핸 -2.5 가능 구간)
    y_true_30 = (home_wins['real_set_diff'] == 3).astype(int)
    best_th_30 = 0
    best_f1_30 = 0
    
    for th in range(5, 25): 
        y_pred = (home_wins['pred_point_diff'] >= th).astype(int)
        score = f1_score(y_true_30, y_pred, zero_division=0)
        if score > best_f1_30:
            best_f1_30 = score
            best_th_30 = th
            
    print(f"   🏆 [3:0 셧아웃] 최적 기준: +{best_th_30}점 이상")
    print(f"      (이 점수 이상일 때 -2.5 마핸 성공률 급상승)")

    # (2) 3:2 접전 (플핸 +1.5 필수 구간)
    y_true_32 = (home_wins['real_set_diff'] == 1).astype(int)
    best_th_32 = 0
    best_f1_32 = 0
    
    for th in range(1, 15):
        y_pred = (home_wins['pred_point_diff'] <= th).astype(int)
        score = f1_score(y_true_32, y_pred, zero_division=0)
        if score > best_f1_32:
            best_f1_32 = score
            best_th_32 = th
            
    print(f"   🏆 [3:2 풀세트] 최적 기준: +{best_th_32}점 미만")
    print(f"      (이 점수 미만일 때 마핸 절대 금지)")

if __name__ == "__main__":
    analyze_point_thresholds()