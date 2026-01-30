import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os

# =========================================================
# 1. 공통 데이터 준비 (전처리)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")

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

def compare_models():
    print("⚖️ [최종 검증] 기존 모델 vs 통계적 개선 모델 비교\n")

    # 1. 데이터 로드
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("❌ 데이터 파일 없음")
        return

    if 'set_score' in df.columns: df.rename(columns={'set_score': 'score'}, inplace=True)
    if 'team_name' in df.columns: df.rename(columns={'team_name': 'tsname'}, inplace=True)
    
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df = df.sort_values(['game_date', 'game_num'])

    # 숫자 변환
    for c in ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 팀별 집계
    team_grp = df.groupby(['game_date', 'game_num', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'home_team': 'first', 'score': 'first'
    }).reset_index()

    team_grp['attack_rate'] = team_grp.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_grp['receive_rate'] = team_grp.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    team_grp['is_home'] = team_grp.apply(lambda r: r['team_std'] == get_standardized_name(r['home_team']), axis=1)

    # 승패 타겟
    def check_win(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if len(s)<2: return 0
            my, opp = (s[0], s[1]) if row['is_home'] else (s[1], s[0])
            return 1 if my > opp else 0
        except: return 0
    team_grp['is_win'] = team_grp.apply(check_win, axis=1)

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
            'result_win': h['is_win']
        })

        w_h = h['is_win']
        exp_h = 1 / (1 + 10 ** ((elo[ta] - elo[th]) / 400))
        elo[th] += 20 * (w_h - exp_h)
        elo[ta] += 20 * ((1 - w_h) - (1 - exp_h))

    train_df = pd.DataFrame(matches).dropna()
    if len(train_df) == 0: return

    # =========================================================
    # 🥊 모델 비교 시작
    # =========================================================
    scaler = StandardScaler()
    y = train_df['result_win']

    # --- Model A: 기존 모델 (단순 변수 투입) ---
    features_a = ['diff_elo', 'diff_att', 'diff_block', 'diff_serve', 'diff_recv', 'diff_fault']
    X_a = train_df[features_a].copy()
    X_a['diff_fault'] = -X_a['diff_fault'] # 범실 반전
    X_a_scaled = scaler.fit_transform(X_a)
    
    model_a = LogisticRegression(C=1.0)
    model_a.fit(X_a_scaled, y)
    acc_a = np.mean(cross_val_score(model_a, X_a_scaled, y, cv=5))

    # --- Model B: 신규 모델 (직교화 + 상호작용) ---
    # 1. 직교화 (Pure Attack 추출)
    reg = LinearRegression()
    reg.fit(train_df[['diff_elo']], train_df['diff_att'])
    pure_att = train_df['diff_att'] - reg.predict(train_df[['diff_elo']])
    
    # 2. 상호작용 (ELO * Attack)
    inter_elo_att = train_df['diff_elo'] * train_df['diff_att'] / 1000

    X_b = pd.DataFrame({
        'diff_elo': train_df['diff_elo'],
        'pure_att': pure_att,          # [핵심] ELO 영향력 제거된 순수 공격력
        'inter_elo_att': inter_elo_att, # [핵심] 강팀간 대결 변수
        'diff_block': train_df['diff_block'],
        'diff_serve': train_df['diff_serve'],
        'diff_recv': train_df['diff_recv'],
        'diff_fault_inv': -train_df['diff_fault']
    })
    X_b_scaled = scaler.fit_transform(X_b)
    
    model_b = LogisticRegression(C=1.0)
    model_b.fit(X_b_scaled, y)
    acc_b = np.mean(cross_val_score(model_b, X_b_scaled, y, cv=5))

    # =========================================================
    # 📊 결과 리포트
    # =========================================================
    print(f"📊 [정확도 비교] (5-Fold CV)")
    print(f"1️⃣ 기존 모델 (단순 합): {acc_a*100:.2f}%")
    print(f"2️⃣ 신규 모델 (통계 기법): {acc_b*100:.2f}%")
    
    print("\n🔍 [설명력(가중치) 비교 - 공격력 부호 확인]")
    
    # Model A 가중치
    att_idx_a = features_a.index('diff_att')
    weight_a = model_a.coef_[0][att_idx_a]
    
    # Model B 가중치 (Pure Att)
    att_idx_b = list(X_b.columns).index('pure_att')
    weight_b = model_b.coef_[0][att_idx_b]
    
    print(f"1️⃣ 기존 모델 '공격력' 가중치: {weight_a:.4f} {'❌ (음수 위험)' if weight_a < 0 else '✅'}")
    print(f"2️⃣ 신규 모델 '순수 공격력' 가중치: {weight_b:.4f} {'❌' if weight_b < 0 else '✅ (정상 양수)'}")
    
    if weight_a < 0 and weight_b > 0:
        print("\n✨ 결론: 신규 모델이 '다중공선성' 문제를 완벽하게 해결했습니다!")
        print("   (기존 모델은 ELO 때문에 공격력을 깎아먹었지만, 신규 모델은 공격력을 올바르게 평가함)")
    elif acc_b > acc_a:
        print("\n✨ 결론: 신규 모델이 예측 정확도 면에서 더 우수합니다.")
    else:
        print("\n📝 결론: 성능 차이는 미미하지만, 신규 모델이 통계적으로 더 건전합니다.")

if __name__ == "__main__":
    compare_models()