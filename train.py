import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# =========================================================
# 1. 설정 및 데이터 로드
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")
MODEL_FILE = os.path.join(BASE_DIR, "kovo_dual_model.pkl")

def get_standardized_name(name):
    """ 팀명 표준화: 4번/5번 동일 적용 필수 """
    if pd.isna(name): return ""
    # 공백 제거 및 대문자 변환
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
    return name_str # 매핑 안되면 원본(공백제거) 리턴

def train_best_model():
    print("🚀 Step 4: [Final] 모델 학습 (이름표 동기화 + Rolling Mean)")

    if not os.path.exists(DATA_FILE):
        print(f"❌ 데이터 파일 없음: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    
    # [중요] 컬럼명 안전장치
    if 'set_score' in df.columns: df.rename(columns={'set_score': 'score'}, inplace=True)
    if 'team_name' in df.columns: df.rename(columns={'team_name': 'tsname'}, inplace=True)

    # [중요] 팀 이름 문자열 강제 변환 (에러 방지)
    df['tsname'] = df['tsname'].astype(str)

    df['game_date'] = pd.to_datetime(df['game_date'])
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df = df.sort_values(['game_date', 'game_num'])

    # 숫자 변환
    for c in ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt', 'point']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 팀별 집계
    team_grp = df.groupby(['game_date', 'game_num', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'home_team': 'first', 'score': 'first'
    }).reset_index()

    team_grp['attack_rate'] = team_grp.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_grp['receive_rate'] = team_grp.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    
    # 홈팀 이름도 표준화해서 비교
    team_grp['home_team_std'] = team_grp['home_team'].apply(get_standardized_name)
    team_grp['is_home'] = team_grp['team_std'] == team_grp['home_team_std']

    def check_win_diff(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if len(s)<2: return 0, 0
            my, opp = (s[0], s[1]) if row['is_home'] else (s[1], s[0])
            return (1 if my > opp else 0), (my - opp)
        except: return 0, 0
    team_grp[['is_win', 'set_diff']] = team_grp.apply(lambda r: pd.Series(check_win_diff(r)), axis=1)

    # Rolling Mean (5경기)
    team_grp = team_grp.sort_values(['team_std', 'game_date'])
    metrics = ['attack_rate', 'bs', 'ss', 'err', 'receive_rate']
    
    for m in metrics:
        team_grp[f'roll_{m}'] = team_grp.groupby('team_std')[m].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    elo = {t: 1500 for t in team_grp['team_std'].unique()}
    matches = []

    sorted_games = team_grp.sort_values(['game_date', 'game_num'])
    
    for _, grp in sorted_games.groupby(['game_date', 'game_num']):
        if len(grp) != 2: continue
        
        # 홈/어웨이 구분 안전장치
        h_rows = grp[grp['is_home'] == True]
        a_rows = grp[grp['is_home'] == False]
        if h_rows.empty or a_rows.empty: continue
        
        h, a = h_rows.iloc[0], a_rows.iloc[0]
        th, ta = h['team_std'], a['team_std']
        
        matches.append({
            'diff_elo': elo[th] - elo[ta],
            'diff_att': h['roll_attack_rate'] - a['roll_attack_rate'],
            'diff_block': h['roll_bs'] - a['roll_bs'],
            'diff_serve': h['roll_ss'] - a['roll_ss'],
            'diff_recv': h['roll_receive_rate'] - a['roll_receive_rate'],
            'diff_fault': h['roll_err'] - a['roll_err'],
            'result_win': h['is_win'],
            'result_diff': h['set_diff']
        })
        
        w_h = h['is_win']
        exp_h = 1 / (1 + 10 ** ((elo[ta] - elo[th]) / 400))
        elo[th] += 20 * (w_h - exp_h)
        elo[ta] += 20 * ((1 - w_h) - (1 - exp_h))

    train_df = pd.DataFrame(matches).dropna()
    
    # 학습 준비
    train_df['diff_fault'] = -train_df['diff_fault'] # 범실 반전
    features = ['diff_elo', 'diff_att', 'diff_block', 'diff_serve', 'diff_recv', 'diff_fault']
    
    X = train_df[features]
    y = train_df['result_win']
    y_reg = train_df['result_diff']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_scaled, y)
    
    reg_model = Ridge(alpha=1.0)
    reg_model.fit(X_scaled, y_reg)

    save_pkg = {
        'classifier': clf,
        'regressor': reg_model,
        'scaler': scaler,
        'features': features,
        'is_advanced': False, 
        'use_ewma': False
    }
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_pkg, f)
    print(f"💾 모델 저장 완료 (IBK/도로공사 매핑 적용됨): {MODEL_FILE}")

if __name__ == "__main__":
    train_best_model()