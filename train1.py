import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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

def train_best_model():
    print("🚀 Step 4: [Final] 모델 학습 (논리 제약 적용 Ver.)")

    if not os.path.exists(DATA_FILE):
        print(f"❌ 데이터 파일 없음: {DATA_FILE}")
        return

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

    def check_win(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if len(s)<2: return 0
            my, opp = (s[0], s[1]) if row['is_home'] else (s[1], s[0])
            return 1 if my > opp else 0
        except: return 0
    team_grp['is_win'] = team_grp.apply(check_win, axis=1)

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
        point_diff = h['point'] - a['point']

        matches.append({
            'diff_elo': elo[th] - elo[ta],
            'diff_att': h['roll_attack_rate'] - a['roll_attack_rate'],
            'diff_block': h['roll_bs'] - a['roll_bs'],
            'diff_serve': h['roll_ss'] - a['roll_ss'],
            'diff_recv': h['roll_receive_rate'] - a['roll_receive_rate'],
            'diff_fault': h['roll_err'] - a['roll_err'],
            'result_win': h['is_win'],
            'point_diff': point_diff
        })
        
        w_h = h['is_win']
        exp_h = 1 / (1 + 10 ** ((elo[ta] - elo[th]) / 400))
        elo[th] += 20 * (w_h - exp_h)
        elo[ta] += 20 * ((1 - w_h) - (1 - exp_h))

    train_df = pd.DataFrame(matches).dropna()
    
    # [중요] 범실은 적을수록 좋으므로 부호 반전
    # 반전 후 양수 가중치 = 감점 효과 (Penalty)
    train_df['diff_fault'] = -train_df['diff_fault']
    
    features = ['diff_elo', 'diff_att', 'diff_block', 'diff_serve', 'diff_recv', 'diff_fault']
    
    X = train_df[features]
    y = train_df['result_win']
    y_reg = train_df['point_diff']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    # ---------------------------------------------------------
    # 🤖 [핵심] 논리 제약 (Positive Constraint) 적용
    # ---------------------------------------------------------
    print("🔍 모델 학습 중... (Positive=True 적용)")
    print("   👉 ELO, 공격, 블로킹, 서브 등 모든 지표가 '양수 효과'를 내도록 강제합니다.")
    print("   👉 범실은 입력값을 뒤집었으므로, 가중치가 양수면 '감점'이 됩니다.")

    param_grid = {'alpha': [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]}
    
    # positive=True: 계수를 무조건 0 이상으로 만듦
    # 이렇게 하면 엉뚱하게 '공격 잘하면 손해' 같은 결과가 절대 안 나옴.
    estimator = Ridge(positive=True)
    
    grid_reg = GridSearchCV(estimator, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_reg.fit(X_scaled, y_reg)
    
    best_reg = grid_reg.best_estimator_
    print(f"   👉 최적 Alpha: {best_reg.alpha}")
    
    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_scaled, y)

    save_pkg = {
        'classifier': clf,
        'regressor': best_reg,
        'scaler': scaler,
        'features': features,
        'is_advanced': False
    }
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_pkg, f)
    print(f"💾 모델 재학습 완료: {MODEL_FILE}")

if __name__ == "__main__":
    train_best_model()