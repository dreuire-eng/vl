import pandas as pd
import numpy as np
import pickle
import sys
from datetime import datetime, timedelta

# =========================================================
# 1. 설정 및 유틸리티
# =========================================================
HISTORY_FILE = "kovo_analysis_ready.csv"   # 03번 결과물
SCHEDULE_FILE = "kovo_schedule_result.csv" # 01번 결과물
MODEL_FILE = "kovo_dual_model.pkl"         # 04번 결과물

def get_standardized_name(name):
    """ 팀명 표준화 """
    if pd.isna(name): return ""
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

# =========================================================
# 2. 현재 팀 상태(ELO, 최근스탯) 재구축
# =========================================================
def build_current_team_stats():
    try:
        df = pd.read_csv(HISTORY_FILE)
    except FileNotFoundError:
        print(f"❌ {HISTORY_FILE} 파일이 없습니다. 03번을 먼저 실행하세요.")
        sys.exit()

    # 컬럼 이름 강제 통일 (에러 방지용)
    if 'set_score' in df.columns:
        df.rename(columns={'set_score': 'score'}, inplace=True)
    if 'team_name' in df.columns:
        df.rename(columns={'team_name': 'tsname'}, inplace=True)

    # 필수 컬럼 체크
    if 'tsname' not in df.columns or 'score' not in df.columns:
        print(f"🚨 컬럼 누락 에러! 현재 컬럼: {list(df.columns)}")
        sys.exit()

    # 팀명 표준화
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'game_num'])

    # 숫자 변환
    num_cols = ['point', 'ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 팀별 경기 집계
    team_grp = df.groupby(['game_date', 'game_num', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'home_team': 'first', 'score': 'first'
    }).reset_index()

    # 성공률 계산
    team_stats = team_grp.sort_values(['game_date', 'game_num'])
    team_stats['attack_rate'] = team_stats.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_stats['receive_rate'] = team_stats.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    
    # 홈 여부
    team_stats['is_home'] = team_stats.apply(lambda r: r['team_std'] == get_standardized_name(r['home_team']), axis=1)
    
    # 승패 파싱 (ELO 계산용)
    def check_win(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if len(s) < 2: return 0
            my, opp = (s[0], s[1]) if row['is_home'] else (s[1], s[0])
            return 1 if my > opp else 0
        except: return 0
    
    team_stats['is_win'] = team_stats.apply(check_win, axis=1)

    # 상태 추적
    current_state = {} 
    all_teams = team_stats['team_std'].unique()
    for t in all_teams:
        current_state[t] = {'elo': 1500, 'last_date': None, 'stats_history': []}

    # 역사 복기
    for _, grp in team_stats.groupby(['game_date', 'game_num']):
        if len(grp) != 2: continue
        
        h_row = grp[grp['is_home'] == True]
        a_row = grp[grp['is_home'] == False]
        if h_row.empty or a_row.empty: continue
        
        h, a = h_row.iloc[0], a_row.iloc[0]
        th, ta = h['team_std'], a['team_std']

        elo_h = current_state[th]['elo']
        elo_a = current_state[ta]['elo']
        w_h = h['is_win']
        
        # ELO 업데이트
        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        k = 20
        new_elo_h = elo_h + k * (w_h - exp_h)
        new_elo_a = elo_a + k * ((1 - w_h) - (1 - exp_h))
        
        current_state[th]['elo'] = new_elo_h
        current_state[ta]['elo'] = new_elo_a
        current_state[th]['last_date'] = h['game_date']
        current_state[ta]['last_date'] = a['game_date']
        
        stat_cols = ['attack_rate', 'bs', 'ss', 'err', 'receive_rate']
        current_state[th]['stats_history'].append(h[stat_cols].to_dict())
        current_state[ta]['stats_history'].append(a[stat_cols].to_dict())

    return current_state

# =========================================================
# 3. 예측 실행
# =========================================================
def predict_matchups():
    print("🚀 KOVO 승부 예측 (AI Model V3 - 핸디캡 정밀 분석)")
    print("-" * 50)

    # 1. 모델 로드
    try:
        with open(MODEL_FILE, "rb") as f:
            model_pkg = pickle.load(f)
        
        clf = model_pkg['classifier']
        reg = model_pkg['regressor']
        scaler = model_pkg['scaler']
        features = model_pkg['features']
        is_constrained = model_pkg.get('is_constrained', False)
        
        print(f"🤖 AI 모델 로드 완료: {'논리제약 모드' if is_constrained else '일반 모드'}")
    except FileNotFoundError:
        print(f"❌ {MODEL_FILE} 파일이 없습니다. 04번을 실행하세요.")
        return

    # 2. 팀 상태 최신화
    print("🔄 팀 전력 데이터 최신화 중...")
    team_state = build_current_team_stats()

    # 3. 오늘 일정 로드
    sch = pd.read_csv(SCHEDULE_FILE)
    sch['gdate'] = pd.to_datetime(sch['gdate'])
    sch['hname'] = sch['hname'].apply(get_standardized_name)
    sch['aname'] = sch['aname'].apply(get_standardized_name)
    
    today = datetime.now().strftime("%Y-%m-%d")
    # today = "2026-01-18" # 테스트 날짜 필요시 수정
    
    todays_games = sch[sch['gdate'] == today]
    
    if todays_games.empty:
        print(f"📅 {today}: 예정된 경기가 없습니다.")
        return

    print(f"📅 {today} 경기 분석 시작 ({len(todays_games)}경기)\n")

    for _, row in todays_games.iterrows():
        h_team = row['hname']
        a_team = row['aname']
        
        if h_team not in team_state or a_team not in team_state:
            print(f"⚠️ {h_team} vs {a_team}: 데이터 부족")
            continue
            
        st_h = team_state[h_team]
        st_a = team_state[a_team]
        
        # 피처 생성
        diff_elo = st_h['elo'] - st_a['elo']
        
        def get_rest(last_date):
            if pd.isna(last_date): return 4
            return (pd.to_datetime(today) - last_date).days
        
        diff_rest = min(get_rest(st_h['last_date']), 14) - min(get_rest(st_a['last_date']), 14)

        def get_avg_stat(history, key):
            if not history: return 0
            recent = history[-5:]
            vals = [x[key] for x in recent]
            return sum(vals) / len(vals)

        metrics = {'diff_att': 'attack_rate', 'diff_block': 'bs', 'diff_serve': 'ss', 
                   'diff_recv': 'receive_rate', 'diff_fault': 'err'}
        
        input_features = {}
        input_features['diff_elo'] = diff_elo
        input_features['diff_rest'] = diff_rest
        for feat_name, key in metrics.items():
            input_features[feat_name] = get_avg_stat(st_h['stats_history'], key) - get_avg_stat(st_a['stats_history'], key)
            
        X_input = pd.DataFrame([input_features], columns=features)
        X_scaled = pd.DataFrame(scaler.transform(X_input), columns=features)
        
        if is_constrained:
            X_scaled['diff_fault'] = -X_scaled['diff_fault']

        # 예측 수행
        prob_home = clf.predict_proba(X_scaled)[0][1]
        prob_away = 1 - prob_home
        pred_diff = reg.predict(X_scaled)[0]

        # =================================================
        # 🎯 승률 기반 세트 스코어 및 핸디캡 전략 수립
        # =================================================
        if prob_home > 0.5:
            winner = h_team
            p_win = prob_home
            score_diff_sign = "+" # 홈 우세
        else:
            winner = a_team
            p_win = prob_away
            score_diff_sign = "-" # 원정 우세
            
        # 확률 구간별 시나리오
        if p_win >= 0.75:
            est_score = "3:0 (셧아웃 유력)"
            risk_level = "낮음"
        elif p_win >= 0.60:
            est_score = "3:1 (우세)"
            risk_level = "중간"
        else:
            est_score = "3:2 (풀세트 초접전)"
            risk_level = "높음"

        # 출력
        print(f"🏐 {h_team} (Home) vs {a_team} (Away)")
        print(f"   📊 전력: ELO {st_h['elo']:.0f} vs {st_a['elo']:.0f} (ELO차이 {diff_elo:+.0f})")
        
        icon = "🏠" if prob_home > 0.5 else "✈️"
        print(f"   🏆 예측 승자: {icon} {winner} (확률 {p_win*100:.1f}%)")
        print(f"   🔢 예상 스코어: {est_score}")
        print(f"   📉 예상 득실차: {pred_diff:+.1f}점 (양수=홈, 음수=원정 우세)")
        
        print("\n   💡 [베팅 가이드]")
        if risk_level == "낮음":
            print(f"      👉 {winner} 마핸승(-1.5) : ✅ 추천 (안전)")
            print(f"      👉 언더/오버 : 🔽 언더 가능성 (셧아웃 예상)")
        elif risk_level == "중간":
            print(f"      👉 {winner} 일반승 : ✅ 추천")
            print(f"      👉 {winner} 마핸승(-1.5) : ⚠️ 조심 (한 세트 내줄 확률 높음)")
            print(f"      👉 언더/오버 : 🟢 오버 추천 (4세트 이상)")
        else:
            print(f"      👉 승패 : 🚫 패스 권장 (너무 박빙)")
            print(f"      👉 핸디캡 : 상대팀 플핸(+1.5) 추천 🍯")
            print(f"      👉 언더/오버 : 🟢 오버 풀매수 (풀세트 예상)")

        print("-" * 50)

if __name__ == "__main__":
    predict_matchups()