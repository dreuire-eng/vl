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

# 남녀 팀 구분 리스트
MEN_TEAMS = ['대한항공', '현대캐피탈', 'KB손해보험', 'OK금융그룹', '한국전력', '우리카드', '삼성화재']
WOMEN_TEAMS = ['흥국생명', '현대건설', '정관장', 'IBK기업은행', 'GS칼텍스', '도로공사', '페퍼저축은행']

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

def get_gender(team_name):
    """ 팀 이름으로 성별 구분 """
    if team_name in MEN_TEAMS: return 'Male'
    if team_name in WOMEN_TEAMS: return 'Female'
    return 'Unknown'

def analyze_by_gender():
    print("🚀 [남녀 구분] 예상 득실차 기반 스코어 분석")
    print("-" * 60)

    if not os.path.exists(MODEL_FILE):
        print("❌ 모델 파일이 없습니다.")
        return
    with open(MODEL_FILE, "rb") as f: pkg = pickle.load(f)
    reg = pkg['regressor'] 
    scaler = pkg['scaler']
    features = pkg['features']

    # 데이터 준비
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
        
        h_rows = grp[grp['is_home'] == True]
        a_rows = grp[grp['is_home'] == False]
        if h_rows.empty or a_rows.empty: continue
            
        h, a = h_rows.iloc[0], a_rows.iloc[0]
        th, ta = h['team_std'], a['team_std']
        
        try:
            s = list(map(int, str(h['score']).split(':')))
            real_set_diff = s[0] - s[1] 
        except: continue

        matches.append({
            'gender': get_gender(th), # 성별 태깅
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

    if not matches: return

    df_m = pd.DataFrame(matches).dropna()
    df_m['diff_fault'] = -df_m['diff_fault']
    
    X = df_m[features]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    df_m['pred_point_diff'] = reg.predict(X_scaled)
    
    # 홈팀 승리 예상 경기만 필터링
    home_wins = df_m[df_m['pred_point_diff'] > 0].copy()
    
    bins = [0, 5, 10, 15, 100]
    labels = ["0~5pts", "5~10pts", "10~15pts", "15pts+"]
    home_wins['bin'] = pd.cut(home_wins['pred_point_diff'], bins=bins, labels=labels)

    # =========================================================
    # 📊 남녀 분리 출력
    # =========================================================
    for gender in ['Male', 'Female']:
        print(f"\n🏐 [{gender}] 리그 분석 결과")
        print(f"{'Pred Points':<15} | {'Games':<5} | {'3:0(%)':<8} | {'3:1(%)':<8} | {'3:2(%)':<8} | {'Fail(%)'}")
        print("-" * 75)
        
        subset_g = home_wins[home_wins['gender'] == gender]
        
        for label in labels:
            subset = subset_g[subset_g['bin'] == label]
            total = len(subset)
            if total == 0: continue
            
            cnt_30 = len(subset[subset['real_set_diff'] == 3])
            cnt_31 = len(subset[subset['real_set_diff'] == 2])
            cnt_32 = len(subset[subset['real_set_diff'] == 1])
            cnt_fail = len(subset[subset['real_set_diff'] < 0])
            
            print(f"{label:<15} | {total:<5} | {cnt_30/total*100:>6.1f}%  | {cnt_31/total*100:>6.1f}%  | {cnt_32/total*100:>6.1f}%  | {cnt_fail/total*100:>6.1f}%")

if __name__ == "__main__":
    analyze_by_gender()