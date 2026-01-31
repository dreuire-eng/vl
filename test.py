import pandas as pd
import numpy as np
import pickle
import os

# =========================================================
# 1. 설정 및 데이터 로드
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")
MODEL_FILE = os.path.join(BASE_DIR, "kovo_dual_model.pkl")

# 남자부만 필터링 (사용자 요청)
MEN_TEAMS = ['대한항공', '현대캐피탈', 'KB손해보험', 'OK금융그룹', '한국전력', '우리카드', '삼성화재']

def get_standardized_name(name):
    if pd.isna(name): return ""
    name_str = str(name).upper().replace(" ", "")
    mapping = {
        '대한항공': ['KOREANAIR', 'JUMBOS', 'KAL', '대한항공'],
        '현대캐피탈': ['HYUNDAICAPITAL', 'SKYWALKERS', '현대캐피탈'],
        'KB손해보험': ['KBSTARS', 'KBINSURANCE', 'LIG', 'KB손해보험'],
        'OK금융그룹': ['OKFINANCIAL', 'OKSAVINGS', 'OKMAN', 'OK금융', '읏맨'],
        '한국전력': ['KEPCO', 'VIXTORM', 'KOREAELECTRIC', '한국전력'],
        '우리카드': ['WOORICARD', 'WOORIWON', '우리카드'],
        '삼성화재': ['SAMSUNG', 'BLUEFANGS', '삼성화재'],
    }
    for std, keys in mapping.items():
        if any(k in name_str for k in keys): return std
    return name_str

def check_clutch_power():
    print("🚀 [남자부] 승률 구간별 '접전(3:2)' 승자 분석")
    print("   (과연 강팀은 접전에서도 살아남는가?)")
    print("-" * 60)

    if not os.path.exists(MODEL_FILE):
        print("❌ 모델 파일 없음")
        return

    with open(MODEL_FILE, "rb") as f: pkg = pickle.load(f)
    clf = pkg['classifier']
    scaler = pkg['scaler']
    features = pkg['features']

    # 데이터 로드
    df = pd.read_csv(DATA_FILE)
    if 'set_score' in df.columns: df.rename(columns={'set_score': 'score'}, inplace=True)
    if 'team_name' in df.columns: df.rename(columns={'team_name': 'tsname'}, inplace=True)
    df['tsname'] = df['tsname'].astype(str)
    
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    
    # 남자부 필터링
    df = df[df['team_std'].isin(MEN_TEAMS)].copy()
    df = df.sort_values(['game_date', 'game_num'])

    # 숫자 변환
    for c in ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt', 'point']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 전처리
    team_grp = df.groupby(['game_date', 'game_num', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'home_team': 'first', 'score': 'first',
        'point': 'sum'
    }).reset_index()

    team_grp['attack_rate'] = team_grp.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_grp['receive_rate'] = team_grp.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    team_grp['home_team_std'] = team_grp['home_team'].astype(str).apply(get_standardized_name)
    team_grp['is_home'] = team_grp['team_std'] == team_grp['home_team_std']

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
        
        # 실제 결과 확인
        try:
            s = list(map(int, str(h['score']).split(':')))
            h_score, a_score = s[0], s[1]
            total_sets = h_score + a_score
            is_full_set = (total_sets == 5) # 3:2 경기만 추출
            winner_is_home = (h_score > a_score)
        except: continue

        matches.append({
            'diff_elo': elo[th] - elo[ta],
            'diff_att': h['roll_attack_rate'] - a['roll_attack_rate'],
            'diff_block': h['roll_bs'] - a['roll_bs'],
            'diff_serve': h['roll_ss'] - a['roll_ss'],
            'diff_recv': h['roll_receive_rate'] - a['roll_receive_rate'],
            'diff_fault': h['roll_err'] - a['roll_err'], # 04번에서 반전했으면 여기서도 로직 맞춤
            'is_full_set': is_full_set,
            'winner_is_home': winner_is_home
        })
        
        w_h = 1 if h_score > a_score else 0
        exp_h = 1 / (1 + 10 ** ((elo[ta] - elo[th]) / 400))
        elo[th] += 20 * (w_h - exp_h)
        elo[ta] += 20 * ((1 - w_h) - (1 - exp_h))

    # 분석
    df_m = pd.DataFrame(matches).dropna()
    df_m['diff_fault'] = -df_m['diff_fault'] # 04번 논리 따라감
    
    # 3:2 접전 경기만 필터링
    full_sets = df_m[df_m['is_full_set'] == True].copy()
    
    X = full_sets[features]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    full_sets['prob_home'] = clf.predict_proba(X_scaled)[:, 1]
    
    # 구간별 분석
    print(f"\n총 3:2 풀세트 경기 수: {len(full_sets)}게임")
    print(f"{'AI Prob':<15} | {'Games':<5} | {'Favorite Win(%)':<15} | {'Underdog Win(%)'}")
    print("-" * 65)
    
    # 1. 진흙탕 구간 (승률 65% 미만) - 홈팀 승률 0.5~0.65
    low_conf = full_sets[(full_sets['prob_home'] >= 0.5) & (full_sets['prob_home'] < 0.65)]
    lc_total = len(low_conf)
    if lc_total > 0:
        lc_win = len(low_conf[low_conf['winner_is_home'] == True])
        print(f"{'50% ~ 65%':<15} | {lc_total:<5} | {lc_win/lc_total*100:>6.1f}% (혼전)    | {(lc_total-lc_win)/lc_total*100:>6.1f}% (역배)")
    
    # 2. 강팀 구간 (승률 65% 이상) - 홈팀 승률 0.65 이상
    high_conf = full_sets[full_sets['prob_home'] >= 0.65]
    hc_total = len(high_conf)
    if hc_total > 0:
        hc_win = len(high_conf[high_conf['winner_is_home'] == True])
        print(f"{'65% +':<15} | {hc_total:<5} | {hc_win/hc_total*100:>6.1f}% (꾸역승)  | {(hc_total-hc_win)/hc_total*100:>6.1f}%")

    print("\n💡 [결론]")
    print("   - 50~65% 구간은 승률이 반반에 가깝다면 -> '운빨 게임' (절대 패스)")
    print("   - 65% 이상 구간에서 승률이 높다면 -> '강팀이 결국 이김' (일반승 축)")

if __name__ == "__main__":
    check_clutch_power()