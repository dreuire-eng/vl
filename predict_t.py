import pandas as pd
import numpy as np
import pickle
import sys
import os
from datetime import datetime

# [추가] 성별 구분 리스트
MEN_TEAMS = ['대한항공', '현대캐피탈', 'KB손해보험', 'OK금융그룹', '한국전력', '우리카드', '삼성화재']
WOMEN_TEAMS = ['흥국생명', '현대건설', '정관장', 'IBK기업은행', 'GS칼텍스', '도로공사', '페퍼저축은행']

def get_gender(team_name):
    if team_name in MEN_TEAMS: return 'Male'
    if team_name in WOMEN_TEAMS: return 'Female'
    return 'Unknown'

# =========================================================
# 1. 설정 및 유틸리티
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")
SCHEDULE_FILE = os.path.join(BASE_DIR, "kovo_schedule_result.csv")
MODEL_FILE = os.path.join(BASE_DIR, "kovo_dual_model.pkl")

def get_standardized_name(name):
    """ 팀명 표준화 (4번과 동일) """
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

# =========================================================
# 2. 현재 팀 상태 재구축
# =========================================================
def build_current_team_stats():
    if not os.path.exists(HISTORY_FILE):
        print(f"❌ {HISTORY_FILE} 파일이 없습니다.")
        sys.exit()

    df = pd.read_csv(HISTORY_FILE)
    if 'set_score' in df.columns: df.rename(columns={'set_score': 'score'}, inplace=True)
    if 'team_name' in df.columns: df.rename(columns={'team_name': 'tsname'}, inplace=True)

    # [핵심] 여기서 강제 통일!
    df['tsname'] = df['tsname'].astype(str)
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df['home_team_std'] = df['home_team'].astype(str).apply(get_standardized_name)
    
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'game_num'])

    for c in ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    team_grp = df.groupby(['game_date', 'game_num', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'home_team': 'first', 'score': 'first'
    }).reset_index()

    team_stats = team_grp.sort_values(['game_date', 'game_num'])
    team_stats['attack_rate'] = team_stats.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_stats['receive_rate'] = team_stats.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)
    # 홈팀 여부 판단 (표준화된 이름으로 비교)
    team_stats['is_home'] = team_stats.apply(lambda r: r['team_std'] == get_standardized_name(r['home_team']), axis=1)
    
    def check_win(row):
        try:
            s = list(map(int, str(row['score']).split(':')))
            if len(s) < 2: return 0
            my, opp = (s[0], s[1]) if row['is_home'] else (s[1], s[0])
            return 1 if my > opp else 0
        except: return 0
    team_stats['is_win'] = team_stats.apply(check_win, axis=1)

    current_state = {} 
    
    # [디버깅] 로드된 팀 목록 확인
    loaded_teams = team_stats['team_std'].unique()
    # print(f"📋 [DEBUG] 파일에서 인식된 팀 목록: {list(loaded_teams)}")
    
    for t in loaded_teams:
        current_state[t] = {'elo': 1500, 'stats_history': []}

    for _, grp in team_stats.groupby(['game_date', 'game_num']):
        if len(grp) != 2: continue
        
        h_rows = grp[grp['is_home']==True]
        a_rows = grp[grp['is_home']==False]
        if h_rows.empty or a_rows.empty: continue
        
        h, a = h_rows.iloc[0], a_rows.iloc[0]
        th, ta = h['team_std'], a['team_std']

        w_h = h['is_win']
        elo_h, elo_a = current_state[th]['elo'], current_state[ta]['elo']
        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        
        current_state[th]['elo'] += 20 * (w_h - exp_h)
        current_state[ta]['elo'] += 20 * ((1 - w_h) - (1 - exp_h))
        
        stat_cols = ['attack_rate', 'bs', 'ss', 'err', 'receive_rate']
        current_state[th]['stats_history'].append(h[stat_cols].to_dict())
        current_state[ta]['stats_history'].append(a[stat_cols].to_dict())

    return current_state

# =========================================================
# 3. 예측 실행 (남녀 구분 로직 적용 Ver.)
# =========================================================
def predict_matchups():
    print("🚀 KOVO 승부 예측 (Gender-Specific Logic)")
    print("-" * 60)

    try:
        with open(MODEL_FILE, "rb") as f: model_pkg = pickle.load(f)
        clf = model_pkg['classifier']
        reg = model_pkg['regressor']
        scaler = model_pkg['scaler']
        features = model_pkg['features']
        print(f"🤖 모델 로드 완료: Point Diff + Gender Split")
    except FileNotFoundError:
        print(f"❌ {MODEL_FILE} 파일이 없습니다.")
        return

    print("🔄 팀 전력 데이터 구축 중...")
    team_state = build_current_team_stats()

    sch = pd.read_csv(SCHEDULE_FILE)
    sch['gdate'] = pd.to_datetime(sch['gdate'])
    sch['hname'] = sch['hname'].apply(get_standardized_name)
    sch['aname'] = sch['aname'].apply(get_standardized_name)
    
    today = datetime.now().strftime("%Y-%m-%d")
    todays_games = sch[sch['gdate'] == today]
    
    if todays_games.empty:
        print(f"📅 {today}: 예정된 경기가 없습니다.")
        return

    print(f"📅 {today} 경기 분석 시작 ({len(todays_games)}경기)\n")

    for idx, row in todays_games.iterrows():
        h_team = row['hname']
        a_team = row['aname']
        
        if h_team not in team_state:
            print(f"⚠️ {h_team} vs {a_team}: 데이터 부족")
            continue
            
        st_h = team_state[h_team]
        st_a = team_state[a_team]
        
        # 성별 확인
        gender = get_gender(h_team)
        
        diff_elo = st_h['elo'] - st_a['elo']
        
        def get_avg(hist, key):
            if not hist: return 0
            recent = hist[-5:]
            vals = [x[key] for x in recent]
            return sum(vals) / len(vals)

        metrics = {'diff_att': 'attack_rate', 'diff_block': 'bs', 'diff_serve': 'ss', 
                   'diff_recv': 'receive_rate', 'diff_fault': 'err'}
        
        input_data = {'diff_elo': diff_elo}
        for feat, key in metrics.items():
            input_data[feat] = get_avg(st_h['stats_history'], key) - get_avg(st_a['stats_history'], key)
            
        df_input = pd.DataFrame([input_data])
        if 'diff_fault' in features: df_input['diff_fault'] = -df_input['diff_fault']

        X_scaled = pd.DataFrame(scaler.transform(df_input[features]), columns=features)
        
        prob_home = clf.predict_proba(X_scaled)[0][1]
        pred_diff = reg.predict(X_scaled)[0] # 예상 득실차
        
        if prob_home > 0.5:
            winner, p_win, loser = h_team, prob_home, a_team
        else:
            winner, p_win, loser = a_team, 1 - prob_home, h_team
            
        abs_diff = abs(pred_diff)

        # =========================================================
        # 🎯 [Final Ver.] 승률 & 득실차 교차 검증 (남녀 차등 + 3:2 리스크 반영)
        # =========================================================
        est_score = ""
        risk = ""
        guide_msg = []

        # ---------------------------------------------------------
        # ♂️ 남자부: 강팀도 5세트 가면 죽는다 (데이터 증명 완료)
        # ---------------------------------------------------------
        if gender == 'Male':
            # 1. [승률 필터] 65% 미만은 믿지 마라 (기존 동일)
            if p_win < 0.65:
                est_score = "3:2 (AI 승률 신뢰도 낮음)"
                risk = "매우 높음"
                guide_msg.append(f"👉 승패 : 🚫 패스 권장 (50:50 동전던지기)")
                guide_msg.append(f"👉 핸디캡 : 🍯 {loser} +1.5 플핸 (역배 45% 터짐)")
                
            # 2. [승률 통과] 65% 이상이지만... 점수차를 봐야 한다
            else:
                if abs_diff >= 10.0: # 완벽한 구간
                    est_score = "3:0 (셧아웃 유력)"
                    risk = "낮음"
                    guide_msg.append(f"👉 {winner} -1.5 마핸 : 💎 강력 추천")
                    
                elif abs_diff >= 7.0: # 일반적인 승리
                    est_score = "3:1 (우세)"
                    risk = "중간"
                    guide_msg.append(f"👉 {winner} 일반승 : ✅ 추천")
                    guide_msg.append(f"👉 {winner} -1.5 마핸 : ⚠️ 소액 접근")
                    
                else: # [핵심 수정] 승률은 높은데 점수차 7점 미만 (3:2 예상)
                    # 데이터: 정배 승률 52.8% vs 역배 47.2% -> 베팅 가치 없음
                    est_score = "3:2 (강팀의 고전 예상)"
                    risk = "높음" 
                    guide_msg.append(f"👉 승패 : 🚫 절대 패스 (이 구간 승률 52% 불과)")
                    guide_msg.append(f"👉 핸디캡 : 🔥 {loser} +1.5 플핸 (무조건 먹는 꿀통)")
                    guide_msg.append(f"👉 언더/오버 : 🟢 오버 (풀세트 혈전)")

        # ---------------------------------------------------------
        # ♀️ 여자부: 물 들어올 때 노 저어라 (기존 동일)
        # ---------------------------------------------------------
        else:
            if abs_diff >= 10.0: 
                est_score = "3:0 (강력한 셧아웃)"
                risk = "매우 낮음"
                guide_msg.append(f"👉 {winner} -1.5 마핸 : 💎 전재산(?).. 강력 추천")
                guide_msg.append(f"👉 {winner} -2.5 마핸 : ✅ 추천")
                
            elif abs_diff >= 5.0: 
                est_score = "3:0 or 3:1 (완승)"
                risk = "낮음"
                guide_msg.append(f"👉 {winner} -1.5 마핸 : ✅ 추천 (안전)")
                guide_msg.append(f"👉 {winner} 일반승 : 💎 보너스 배당")
                
            else: 
                est_score = "3:2 (접전승)"
                risk = "중간"
                guide_msg.append(f"👉 {winner} 일반승 : ✅ 추천 (여자부는 강팀이 결국 이김)")
                guide_msg.append(f"👉 핸디캡 : {loser} +1.5 플핸 (보험용)")

        # [최종 출력]
        gender_icon = "‍♂️" if gender == 'Male' else "‍♀️"
        print(f"🏐 [{gender_icon}] {h_team} (Home) vs {a_team} (Away)")
        print(f"   📊 전력: ELO {st_h['elo']:.0f} vs {st_a['elo']:.0f} (ELO차이 {diff_elo:+.0f})")
        
        icon = "🏠" if prob_home > 0.5 else "✈️"
        print(f"   🏆 예측 승자: {icon} {winner} (확률 {p_win*100:.1f}%)")
        print(f"   🔢 예상 스코어: {est_score}")
        print(f"   📉 예상 득실차: {pred_diff:+.1f}점")
        
        print("\n   💡 [베팅 가이드]")
        for msg in guide_msg:
            print(f"      {msg}")

        print("-" * 60)

if __name__ == "__main__":
    predict_matchups()