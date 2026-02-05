import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

# 설정
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
        'OK금융그룹': ['OKFINANCIAL', 'OKSAVINGS', 'OKMAN', 'OK금융', '읏맨', 'OK', 'OK저축은행'],
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
    MEN = ['대한항공', '현대캐피탈', 'KB손해보험', 'OK금융그룹', '한국전력', '우리카드', '삼성화재']
    WOMEN = ['흥국생명', '현대건설', '정관장', 'IBK기업은행', 'GS칼텍스', '도로공사', '페퍼저축은행']
    if team_name in MEN: return 'Male'
    if team_name in WOMEN: return 'Female'
    return 'Unknown'

def optimize_thresholds(df, clf, reg, scaler, features):
    print("\n" + "="*60)
    print("🧠 [Auto-Tuning] 성별 최적 베팅 기준점 탐색")
    print("="*60)

    best_thresholds = {'Male': {}, 'Female': {}}
    X = df[features]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    
    df = df.copy()
    df['prob_home'] = clf.predict_proba(X_scaled)[:, 1]
    df['pred_diff'] = reg.predict(X_scaled)
    df['gender'] = df['team_std'].apply(get_gender)
    
    y_true = df['result_win']
    
    def check_mahand(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if s[0] == 3 and s[1] <= 1: return 1
            return 0
        except: return 0
    df['is_mahand'] = df.apply(check_mahand, axis=1)

    for gender in ['Male', 'Female']:
        subset = df[df['gender'] == gender]
        if len(subset) < 10: continue

        print(f"\n🔍 [{gender}] 최적화 진행 중...")
        best_f1, best_prob = -1, 0.60
        for p in np.arange(0.55, 0.90, 0.05):
            preds = (subset['prob_home'] >= p).astype(int)
            if preds.sum() == 0: continue
            prec = precision_score(subset['result_win'], preds, zero_division=0)
            f1 = f1_score(subset['result_win'], preds, zero_division=0)
            if prec > 0.65 and f1 > best_f1:
                best_f1, best_prob = f1, p
        print(f"   👉 승무패(Win) 최적 확률 컷: {best_prob:.2f}")

        best_m_f1, best_margin = -1, 7.0
        for m in np.arange(3.0, 16.0, 1.0):
            preds = (subset['pred_diff'] >= m).astype(int)
            if preds.sum() == 0: continue
            prec = precision_score(subset['is_mahand'], preds, zero_division=0)
            f1 = f1_score(subset['is_mahand'], preds, zero_division=0)
            if prec > 0.60 and f1 > best_m_f1:
                best_m_f1, best_margin = f1, m
        print(f"   👉 마핸(Handicap) 최적 득실차 컷: {best_margin:.1f}점")
        
        best_thresholds[gender] = {'prob_safe': best_prob, 'margin_safe': best_margin}
    return best_thresholds

def train_best_model():
    print("🚀 Step 4: [Final] 모델 학습 (변수명 대통합 Ver.)")

    if not os.path.exists(DATA_FILE):
        print(f"❌ 데이터 파일 없음: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    
    # 🚨 [중요] rename 로직 제거! 이미 process.py에서 표준화됨.
    # gdate, seasonCode, tsname, hname, aname, score, gnum 사용
    
    df['gdate'] = pd.to_datetime(df['gdate'])
    df['seasonCode'] = df['seasonCode'].astype(str)
    df['tsname'] = df['tsname'].astype(str)
    
    # 팀명 표준화
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df = df.sort_values(['gdate', 'gnum'])

    for c in ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt', 'point']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 시즌 필터링
    unique_seasons = sorted(df['seasonCode'].unique())
    current_season = unique_seasons[-1]
    recent_3_seasons = unique_seasons[-3:]
    print(f"📅 학습 시즌: {recent_3_seasons}, 분석 시즌: {current_season}")

    # 전처리 (game_date -> gdate, game_num -> gnum, home_team -> hname)
    team_grp = df.groupby(['gdate', 'gnum', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'hname': 'first', 'score': 'first',
        'point': 'sum', 'seasonCode': 'first'
    }).reset_index()

    team_grp['attack_rate'] = team_grp.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_grp['receive_rate'] = team_grp.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    
    team_grp['hname_std'] = team_grp['hname'].astype(str).apply(get_standardized_name)
    team_grp['is_home'] = team_grp['team_std'] == team_grp['hname_std']

    def check_win(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if len(s)<2: return 0
            my, opp = (s[0], s[1]) if row['is_home'] else (s[1], s[0])
            return 1 if my > opp else 0
        except: return 0
    team_grp['is_win'] = team_grp.apply(check_win, axis=1)

    metrics = ['attack_rate', 'bs', 'ss', 'err', 'receive_rate']
    team_grp = team_grp.sort_values(['team_std', 'gdate'])
    for m in metrics:
        team_grp[f'roll_{m}'] = team_grp.groupby('team_std')[m].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    elo = {t: 1500 for t in team_grp['team_std'].unique()}
    matches = []
    sorted_games = team_grp.sort_values(['gdate', 'gnum'])
    
    for _, grp in sorted_games.groupby(['gdate', 'gnum']):
        if len(grp) != 2: continue
        h_rows = grp[grp['is_home'] == True]
        a_rows = grp[grp['is_home'] == False]
        if h_rows.empty or a_rows.empty: continue
        
        h, a = h_rows.iloc[0], a_rows.iloc[0]
        th, ta = h['team_std'], a['team_std']
        
        matches.append({
            'seasonCode': h['seasonCode'],
            'team_std': th,
            'score': h['score'],
            'diff_elo': elo[th] - elo[ta],
            'diff_att': h['roll_attack_rate'] - a['roll_attack_rate'],
            'diff_block': h['roll_bs'] - a['roll_bs'],
            'diff_serve': h['roll_ss'] - a['roll_ss'],
            'diff_recv': h['roll_receive_rate'] - a['roll_receive_rate'],
            'diff_fault': h['roll_err'] - a['roll_err'],
            'result_win': h['is_win'],
            'point_diff': h['point'] - a['point']
        })
        
        w_h = h['is_win']
        exp_h = 1 / (1 + 10 ** ((elo[ta] - elo[th]) / 400))
        elo[th] += 20 * (w_h - exp_h)
        elo[ta] += 20 * ((1 - w_h) - (1 - exp_h))

    all_matches_df = pd.DataFrame(matches).dropna()
    all_matches_df['diff_fault'] = -all_matches_df['diff_fault']
    
    train_df = all_matches_df[all_matches_df['seasonCode'].isin(recent_3_seasons)].copy()
    features = ['diff_elo', 'diff_att', 'diff_block', 'diff_serve', 'diff_recv', 'diff_fault']
    
    X = train_df[features]
    y_reg = train_df['point_diff']
    y_clf = train_df['result_win']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    print("🔍 Main Model 학습 중... (Positive=True)")
    grid_reg = GridSearchCV(Ridge(positive=True), {'alpha': [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]}, cv=5)
    grid_reg.fit(X_scaled, y_reg)
    best_reg = grid_reg.best_estimator_
    
    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_scaled, y_clf)
    
    best_thresholds = optimize_thresholds(train_df, clf, best_reg, scaler, features)

    print("\n" + "="*60)
    print(f"📊 시즌 메타 분석 리포트 (현재 시즌: {current_season})")
    print("="*60)
    trend_df = all_matches_df[all_matches_df['seasonCode'] == current_season].copy()
    if len(trend_df) > 10:
        trend_reg = Ridge(positive=True, alpha=best_reg.alpha) 
        trend_reg.fit(pd.DataFrame(scaler.transform(trend_df[features]), columns=features), trend_df['point_diff'])
        
        print(f"{'Feature':<15} | {'3-Year Avg':<10} | {'This Season':<11} | {'Status'}")
        print("-" * 60)
        coef_data = []
        for name, mc, tc in zip(features, best_reg.coef_, trend_reg.coef_):
            coef_data.append({'name': name, 'main': mc, 'trend': tc})
        coef_data.sort(key=lambda x: abs(x['main']), reverse=True)
        for item in coef_data:
            diff = item['trend'] - item['main']
            status = "🔥 급상승" if diff > 0.1 else ("📉 하락" if diff < -0.1 else "➡️ 유지")
            print(f"{item['name']:<15} | {item['main']:>10.4f} | {item['trend']:>11.4f} | {status}")
        print("-" * 60)

    save_pkg = {
        'classifier': clf, 'regressor': best_reg, 'scaler': scaler, 'features': features,
        'thresholds': best_thresholds, 'is_advanced': True
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_pkg, f)
    print(f"\n💾 모델 저장 완료: {MODEL_FILE}")

if __name__ == "__main__":
    train_best_model()