import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import sys
import os

# =========================================================
# 1. 공통 전처리 로직 (그대로 유지)
# =========================================================
def get_standardized_name(name):
    name_upper = str(name).upper().replace(" ", "")
    mapping = {
        '대한항공': ['KOREANAIR', 'JUMBOS', '대한항공', '점보스', 'KAL'],
        '현대캐피탈': ['HYUNDAICAPITAL', 'SKYWALKERS', '현대캐피탈', '스카이워커스'],
        'KB손해보험': ['KBSTARS', 'KBINSURANCE', 'LIG', 'KB손해보험', '케이비'],
        'OK금융그룹': ['OKFINANCIAL', 'OKSAVINGS', 'OKMAN', 'OK금융', '읏맨'],
        '한국전력': ['KEPCO', 'VIXTORM', 'KOREAELECTRIC', '한국전력', '빅스톰'],
        '우리카드': ['WOORICARD', 'WOORIWON', '우리카드', '위비'],
        '삼성화재': ['SAMSUNG', 'BLUEFANGS', '삼성화재', '블루팡스'],
        '흥국생명': ['HEUNGKUK', 'PINKSPIDERS', '흥국생명', '핑크스파이더스'],
        '현대건설': ['HYUNDAIE&C', 'HILLSTATE', '현대건설', '힐스테이트'],
        '정관장': ['JUNGKWANJANG', 'REDSPARKS', 'KGC', 'GINSENG', '정관장'],
        'IBK기업은행': ['IBK', 'ALTOS', 'INDUSTRIALBANK', '기업은행'],
        'GS칼텍스': ['GSCALTEX', 'KIXX', 'GS칼텍스', '킥스'],
        '도로공사': ['HIPASS', 'EXPRESSWAY', '도로공사', '하이패스'],
        '페퍼저축은행': ['PEPPER', 'AIPEPPERS', '페퍼저축은행', '페퍼']
    }
    for std, keys in mapping.items():
        if any(k in name_upper for k in keys): return std
    return name

def train_logic_constrained_model_v2():
    print("🚀 Step 4-2: [논리 제약] 물리적 정합성(Physics-Informed) 강제 학습 (v2)...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 1. 데이터 준비
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "kovo_analysis_ready.csv"))
    except FileNotFoundError:
        print("❌ 'kovo_analysis_ready.csv' 파일이 없습니다.")
        return

    # [수정 1] 컬럼 이름 통일 (set_score -> score)
    # 03번 코드에서 'set_score'로 저장했으므로, 여기서 이름을 'score'로 바꿔줘야 뒤탈이 없습니다.
    if 'set_score' in df.columns:
        df.rename(columns={'set_score': 'score'}, inplace=True)

    # [수정 2] 컬럼 이름 통일 (team_name -> tsname)
    # 03번 코드나 원본에 따라 이름이 다를 수 있어 안전장치 추가
    if 'team_name' in df.columns:
         df.rename(columns={'team_name': 'tsname'}, inplace=True)

    # 필수 컬럼 체크
    if 'tsname' not in df.columns or 'score' not in df.columns:
        print(f"🚨 컬럼 누락 에러! 현재 컬럼: {list(df.columns)}")
        print("   -> 'tsname'(또는 team_name)과 'score'(또는 set_score)가 있어야 합니다.")
        return

    # 날짜 정렬
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(by=['game_date', 'game_num'])
    
    # 팀 이름 표준화
    df['team_std'] = df['tsname'].apply(get_standardized_name)

    # [개선 1] 팀 스탯 집계 방식 변경 (단순 평균 -> 합계 기반 재계산)
    num_cols = ['point', 'attackSuccessRate', 'ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt']
    for c in num_cols:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # 2) 경기별/팀별 합계 계산
    # 이제 'score' 컬럼이 확실히 있으므로 에러가 나지 않습니다.
    team_grp = df.groupby(['game_date', 'game_num', 'team_std'])
    
    team_stats = team_grp.agg({
        'point': 'sum',
        'ats': 'sum',   
        'att': 'sum',   
        'bs': 'sum',    
        'ss': 'sum',    
        'err': 'sum',   
        'rs': 'sum',    
        'rt': 'sum',    
        'home_team': 'first',
        'score': 'first'  # [확인] 위에서 set_score를 score로 바꿨으므로 OK
    }).reset_index()

    # 3) 진짜 팀 성공률 계산 (Weighted Rate)
    team_stats['attack_rate'] = team_stats.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_stats['receive_rate'] = team_stats.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    
    # 홈/어웨이 구분
    team_stats['is_home_check'] = team_stats.apply(
        lambda r: r['team_std'] == get_standardized_name(r['home_team']), axis=1
    )

    # 정렬
    team_stats = team_stats.sort_values(['team_std', 'game_date'])

    # 피처 엔지니어링 (이동 평균)
    metrics = ['attack_rate', 'bs', 'ss', 'err', 'receive_rate']
    
    for m in metrics:
        team_stats[f'roll_{m}'] = team_stats.groupby('team_std')[m].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    # 휴식일 계산
    team_stats['prev_date'] = team_stats.groupby('team_std')['game_date'].shift(1)
    team_stats['rest_days'] = (team_stats['game_date'] - team_stats['prev_date']).dt.days.fillna(4).clip(upper=14)

    # 타겟(승패) 파싱
    def parse_target(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if len(s) < 2: return pd.Series([None, None])
            
            my, opp = (s[0], s[1]) if row['is_home_check'] else (s[1], s[0])
            return pd.Series([1 if my > opp else 0, my - opp])
        except: return pd.Series([None, None])
    
    team_stats[['is_win', 'set_diff']] = team_stats.apply(parse_target, axis=1)
    team_stats = team_stats.dropna(subset=['is_win'])

    # ELO 및 매치업 데이터 생성
    elo = {t: 1500 for t in team_stats['team_std'].unique()}
    matches = []
    
    sorted_games = team_stats.sort_values(['game_date', 'game_num'])
    
    for _, grp in sorted_games.groupby(['game_date', 'game_num']):
        if len(grp) != 2: continue
        
        h_row = grp[grp['is_home_check'] == True]
        a_row = grp[grp['is_home_check'] == False]
        if h_row.empty or a_row.empty: continue
        
        h, a = h_row.iloc[0], a_row.iloc[0]
        th, ta = h['team_std'], a['team_std']
        
        matches.append({
            'diff_elo': elo[th] - elo[ta],
            'diff_rest': h['rest_days'] - a['rest_days'],
            'diff_att': h['roll_attack_rate'] - a['roll_attack_rate'],
            'diff_block': h['roll_bs'] - a['roll_bs'],
            'diff_serve': h['roll_ss'] - a['roll_ss'],
            'diff_recv': h['roll_receive_rate'] - a['roll_receive_rate'],
            'diff_fault': h['roll_err'] - a['roll_err'], 
            'result_win': h['is_win'],
            'result_diff': h['set_diff']
        })

        # ELO 업데이트
        w_h = h['is_win']
        exp_h = 1 / (1 + 10 ** ((elo[ta] - elo[th]) / 400))
        k_factor = 20
        elo[th] += k_factor * (w_h - exp_h)
        elo[ta] += k_factor * ((1 - w_h) - (1 - exp_h))

    # 학습 데이터 준비
    train_df = pd.DataFrame(matches).dropna()
    
    features = ['diff_elo', 'diff_att', 'diff_block', 'diff_serve', 'diff_recv', 'diff_fault']
    
    X = train_df[features]
    y = train_df['result_win']
    y_reg = train_df['result_diff']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    print(f"\n📊 학습 데이터: {len(X)} 경기")
    
    # -------------------------------------------------------------------------
    # 🔥 [핵심] 논리적 제약이 걸린 모델 학습 (Positive Constraints)
    # -------------------------------------------------------------------------
    print("🔍 논리적 가중치 강제 학습 중...")
    
    # 범실 부호 반전
    X_scaled_constrained = X_scaled.copy()
    X_scaled_constrained['diff_fault'] = -X_scaled_constrained['diff_fault'] 
    
    best_model = None
    best_score = 0
    
    c_params = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    
    for c in c_params:
        clf = LogisticRegression(C=c, fit_intercept=True)
        clf.fit(X_scaled_constrained, y)
        
        coefs = clf.coef_[0]
        # 모든 계수가 0보다 큰지 확인 (관용 0.0)
        if np.all(coefs >= -0.001): 
            score = np.mean(cross_val_score(clf, X_scaled_constrained, y, cv=5))
            if score > best_score:
                best_score = score
                best_model = clf
    
    if best_model:
        print(f"🏆 Best Model Found (Acc: {best_score*100:.2f}%)")
        print("   [가중치 분석 - 클수록 승리에 기여]")
        for f, w in zip(features, best_model.coef_[0]):
            real_w = w if f != 'diff_fault' else -w 
            print(f"   - {f}: {real_w:.4f}")
            
        reg_model = Ridge(alpha=1.0)
        reg_model.fit(X_scaled, y_reg)

        with open(os.path.join(BASE_DIR, "kovo_dual_model.pkl"), "wb") as f:
            pickle.dump({
                'classifier': best_model,
                'regressor': reg_model,
                'scaler': scaler,
                'features': features,
                'is_constrained': True
            }, f)
        print("\n💾 모델 저장 완료 (kovo_dual_model.pkl)")
        
    else:
        print("🚨 논리적 정합성을 만족하는 모델을 찾지 못했습니다.")
        clf = LogisticRegression()
        clf.fit(X_scaled, y)
        with open("kovo_dual_model.pkl", "wb") as f:
             pickle.dump({'classifier': clf, 'regressor': Ridge().fit(X_scaled, y_reg), 
                          'scaler': scaler, 'features': features, 'is_constrained': False}, f)
    # -------------------------------------------------------------
    # 🧪 [보너스] 최적의 스코어 임계값(Threshold) 찾기
    # -------------------------------------------------------------
    print("\n🔍 [Grid Search] 최적의 스코어 구분 기준 탐색...")
    
    # 모델 예측 확률 (Training set 기준이지만 경향성 파악엔 충분)
    probs = best_model.predict_proba(X_scaled_constrained)[:, 1] # 홈 승리 확률
    
    # 실제 스코어 차이 (3, 2, 1, -1, -2, -3)
    # y_reg는 '점수차'가 아니라 '세트차'를 예측하도록 학습했어야 더 좋았겠지만,
    # 여기서는 y_reg(점수차) 대신 원본 데이터의 'result_diff'를 사용해야 함.
    # 하지만 train_df가 있으므로 거기서 가져옵니다.
    
    actual_set_diffs = train_df['result_diff'].abs() # 3, 2, 1 (세트 차이는 아니고 점수차라 부정확할 수 있음)
    # 정확히 하려면 04번 데이터 수집 단계에서 '세트 스코어(3:0 등)'를 별도 컬럼으로 저장했어야 합니다.
    # 지금은 '승률 분포'만 찍어보겠습니다.
    
    results = pd.DataFrame({
        'prob': probs,
        'win': y,
        'score_diff': y_reg # 점수차
    })
    
    # 승리한 경기만 추출 (확률 0.5 이상인 경우)
    wins = results[results['win'] == 1]
    
    # 점수차(score_diff)가 클수록 3:0일 확률이 높음.
    # 점수차 분위수(Quantile)로 역추적
    
    # 상위 30% 점수차인 경기들의 평균 승률 -> 3:0 기준
    # 하위 30% 점수차인 경기들의 평균 승률 -> 3:2 기준
    
    t_shutout = wins[wins['score_diff'] >= wins['score_diff'].quantile(0.7)]['prob'].mean()
    t_close = wins[wins['score_diff'] <= wins['score_diff'].quantile(0.3)]['prob'].mean()
    
    print(f"   📊 데이터 분석 결과:")
    print(f"      - 압승(3:0) 경기들의 평균 승률: {t_shutout*100:.1f}%")
    print(f"      - 접전(3:2) 경기들의 평균 승률: {t_close*100:.1f}%")
    
    # 중간값으로 기준 설정
    suggest_t2 = (t_shutout + 0.60) / 2 # 보정
    suggest_t1 = (t_close + 0.50) / 2
    
    print(f"   💡 추천 임계값: {suggest_t1*100:.0f}% (3:2 vs 3:1) / {suggest_t2*100:.0f}% (3:1 vs 3:0)")
if __name__ == "__main__":
    train_logic_constrained_model_v2()