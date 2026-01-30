import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# =========================================================
# 1. 설정 및 유틸리티
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")
MODEL_FILE = os.path.join(BASE_DIR, "kovo_dual_model.pkl")

def get_standardized_name(name):
    if pd.isna(name): return ""
    name_upper = str(name).upper().replace(" ", "")
    mapping = {
        '대한항공': ['KOREANAIR', 'JUMBOS', 'KAL', '대한항공', '점보스'],
        '현대캐피탈': ['HYUNDAICAPITAL', 'SKYWALKERS', '현대캐피탈'],
        'KB손해보험': ['KBSTARS', 'KBINSURANCE', 'LIG', 'KB손해보험'],
        'OK금융그룹': ['OKFINANCIAL', 'OKSAVINGS', 'OKMAN', 'OK금융', '읏맨'],
        '한국전력': ['KEPCO', 'VIXTORM', 'KOREAELECTRIC', '한국전력'],
        '우리카드': ['WOORICARD', 'WOORIWON', '우리카드', '위비'],
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

def train_stats_pro_model():
    print("🚀 Step 4: [통계적 접근] 다중공선성 해결 및 고급 모델링 (Pro Ver.)")

    # 1. 데이터 로드
    if not os.path.exists(DATA_FILE):
        print(f"❌ 데이터 파일 없음: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    if 'set_score' in df.columns: df.rename(columns={'set_score': 'score'}, inplace=True)
    if 'team_name' in df.columns: df.rename(columns={'team_name': 'tsname'}, inplace=True)

    df['game_date'] = pd.to_datetime(df['game_date'])
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df = df.sort_values(['game_date', 'game_num'])

    # 숫자 변환
    cols = ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt', 'point']
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 팀별 집계
    team_grp = df.groupby(['game_date', 'game_num', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'home_team': 'first', 'score': 'first'
    }).reset_index()

    # 파생 변수
    team_grp['attack_rate'] = team_grp.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_grp['receive_rate'] = team_grp.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    team_grp['is_home'] = team_grp.apply(lambda r: r['team_std'] == get_standardized_name(r['home_team']), axis=1)

    # 타겟 설정
    def check_win_diff(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if len(s)<2: return 0, 0
            my, opp = (s[0], s[1]) if row['is_home'] else (s[1], s[0])
            return (1 if my > opp else 0), (my - opp)
        except: return 0, 0

    team_grp[['is_win', 'set_diff']] = team_grp.apply(lambda r: pd.Series(check_win_diff(r)), axis=1)

    # 롤링 스탯 & ELO
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
    
    # =========================================================================
    # 🧪 [Advanced] 다중공선성 해결 및 고급 피처 생성
    # =========================================================================
    print("🔬 고급 통계적 피처 엔지니어링 수행 중...")
    
    # 1. 범실 반전 (음수 -> 양수: 클수록 범실 적어서 좋은 것)
    train_df['diff_fault_inv'] = -train_df['diff_fault'] 

    # 2. 직교화 (Orthogonalization) - ELO 영향 제거한 순수 스탯
    # 
    
    # (1) 순수 공격력
    reg_att = LinearRegression()
    reg_att.fit(train_df[['diff_elo']], train_df['diff_att'])
    train_df['pure_att'] = train_df['diff_att'] - reg_att.predict(train_df[['diff_elo']])

    # (2) 순수 블로킹
    reg_blk = LinearRegression()
    reg_blk.fit(train_df[['diff_elo']], train_df['diff_block'])
    train_df['pure_block'] = train_df['diff_block'] - reg_blk.predict(train_df[['diff_elo']])
    
    # 3. 상호작용 항 (강팀간 대결 변수)
    train_df['inter_elo_att'] = train_df['diff_elo'] * train_df['diff_att'] / 1000 
    
    # 4. 최종 학습 피처 선정
    features = [
        'diff_elo',       # 팀 체급
        'pure_att',       # 순수 공격 폼 (ELO와 독립적)
        'pure_block',     # 순수 블로킹 폼
        'diff_serve',     # 서브
        'diff_recv',      # 리시브
        'diff_fault_inv', # 범실 관리 (반전됨)
        'inter_elo_att'   # 상호작용
    ]
    
    X = train_df[features]
    y = train_df['result_win']
    y_reg = train_df['result_diff']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    # 5. 모델 학습 (L2 규제 적용)
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
    clf.fit(X_scaled, y)
    
    # 검증
    cv_score = np.mean(cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy'))
    print(f"📊 5-Fold 교차검증 정확도: {cv_score*100:.2f}%")
    
    # 가중치 확인
    print("\n🔍 [모델 가중치 분석]")
    for f, w in zip(features, clf.coef_[0]):
        print(f"   - {f}: {w:.4f}")
        
    # 점수차 예측 모델 (Ridge)
    reg_model = Ridge(alpha=1.0)
    reg_model.fit(X_scaled, y_reg)

    # 6. 저장 (직교화 모델 포함)
    save_pkg = {
        'classifier': clf,
        'regressor': reg_model,
        'scaler': scaler,
        'features': features,
        'is_constrained': False, 
        'is_advanced': True,
        'ortho_models': { 'att': reg_att, 'blk': reg_blk }
    }
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(save_pkg, f)
    print(f"\n💾 모델 저장 완료: {MODEL_FILE}")

    # =========================================================================
    # 🧪 [보너스] 최적의 스코어 임계값(Threshold) 찾기
    # =========================================================================
    print("\n🔍 [Grid Search] 최적의 스코어 구분 기준(Threshold) 계산...")
    
    # 학습된 모델로 확률 다시 뽑기
    probs = clf.predict_proba(X_scaled)[:, 1]
    
    analysis_df = pd.DataFrame({
        'prob': probs,
        'win': y,
        'set_diff': y_reg # 실제 세트 득실
    })
    
    # 승리한 경기(홈승)만 분석
    wins = analysis_df[analysis_df['win'] == 1]
    
    # 3:0 승리 (세트득실 3.0에 가까운) vs 3:2 승리 (세트득실 1.0에 가까운)
    # 상위 30% 점수차 -> 셧아웃으로 간주
    # 하위 30% 점수차 -> 접전으로 간주
    
    t_shutout = wins[wins['set_diff'] >= wins['set_diff'].quantile(0.7)]['prob'].mean()
    t_close = wins[wins['set_diff'] <= wins['set_diff'].quantile(0.3)]['prob'].mean()
    
    print(f"   📊 데이터 분석 결과:")
    print(f"      - 셧아웃(3:0) 경기들의 평균 확률: {t_shutout*100:.1f}%")
    print(f"      - 풀세트(3:2) 경기들의 평균 확률: {t_close*100:.1f}%")
    
    # 기준점 잡기 (중간값)
    cut_30 = (t_shutout + 0.60) / 2 # 3:0 기준 (보수적 보정)
    cut_31 = (t_close + 0.50) / 2   # 3:1 기준
    
    print(f"   💡 추천 임계값 적용: {cut_31*100:.0f}% (3:2 구간) / {cut_30*100:.0f}% (3:1 구간)")
    print("-" * 60)

if __name__ == "__main__":
    train_stats_pro_model()