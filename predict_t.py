import pandas as pd
import numpy as np
import pickle
import sys
import os
from datetime import datetime

MEN_TEAMS = ['대한항공', '현대캐피탈', 'KB손해보험', 'OK금융그룹', '한국전력', '우리카드', '삼성화재']
WOMEN_TEAMS = ['흥국생명', '현대건설', '정관장', 'IBK기업은행', 'GS칼텍스', '도로공사', '페퍼저축은행']

def get_gender(team_name):
    if team_name in MEN_TEAMS: return 'Male'
    if team_name in WOMEN_TEAMS: return 'Female'
    return 'Unknown'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")
SCHEDULE_FILE = os.path.join(BASE_DIR, "kovo_schedule_result.csv")
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

def load_model():
    if not os.path.exists(MODEL_FILE): return None
    with open(MODEL_FILE, "rb") as f: return pickle.load(f)

def build_team_stats(df):
    print("🔄 팀 전력 데이터 구축 중 (표준 변수명 적용)...")
    
    # rename 로직 삭제! (이미 process.py에서 표준화됨)
    df['gdate'] = pd.to_datetime(df['gdate'])
    df['team_std'] = df['tsname'].apply(get_standardized_name)
    df = df.sort_values(['gdate', 'gnum'])

    for c in ['ats', 'att', 'bs', 'ss', 'err', 'rs', 'rt', 'point']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # gdate, gnum, hname, score 사용
    team_grp = df.groupby(['gdate', 'gnum', 'team_std']).agg({
        'ats': 'sum', 'att': 'sum', 'bs': 'sum', 'ss': 'sum', 'err': 'sum', 
        'rs': 'sum', 'rt': 'sum', 'hname': 'first', 'score': 'first', 'point': 'sum'
    }).reset_index()

    team_grp['attack_rate'] = team_grp.apply(lambda x: x['ats']/x['att'] if x['att']>0 else 0, axis=1)
    team_grp['receive_rate'] = team_grp.apply(lambda x: x['rs']/x['rt'] if x['rt']>0 else 0, axis=1)

    metrics = ['attack_rate', 'bs', 'ss', 'err', 'receive_rate']
    team_grp = team_grp.sort_values(['team_std', 'gdate'])
    for m in metrics:
        team_grp[f'roll_{m}'] = team_grp.groupby('team_std')[m].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    elo_dict = {t: 1500 for t in team_grp['team_std'].unique()}
    current_elo = {}
    
    sorted_games = team_grp.sort_values(['gdate', 'gnum'])
    
    for _, grp in sorted_games.groupby(['gdate', 'gnum']):
        if len(grp) != 2: continue
        row1, row2 = grp.iloc[0], grp.iloc[1]
        home_std = get_standardized_name(row1['hname'])
        
        if row1['team_std'] == home_std: h, a = row1, row2
        else: h, a = row2, row1
        
        th, ta = h['team_std'], a['team_std']
        try:
            s = list(map(int, str(h['score']).split(':')))
            w_h = 1 if s[0] > s[1] else 0
        except: w_h = 0.5 
        
        exp_h = 1 / (1 + 10 ** ((elo_dict[ta] - elo_dict[th]) / 400))
        elo_dict[th] += 20 * (w_h - exp_h)
        elo_dict[ta] += 20 * ((1 - w_h) - (1 - exp_h))
        current_elo[th], current_elo[ta] = elo_dict[th], elo_dict[ta]

    latest_stats = {}
    last_rows = team_grp.groupby('team_std').last()
    for team in last_rows.index:
        r = last_rows.loc[team]
        latest_stats[team] = {
            'elo': current_elo.get(team, 1500),
            'roll_attack_rate': r['roll_attack_rate'], 'roll_bs': r['roll_bs'],
            'roll_ss': r['roll_ss'], 'roll_err': r['roll_err'], 'roll_receive_rate': r['roll_receive_rate']
        }
    return latest_stats

def predict_matchups():
    pkg = load_model()
    if not pkg: 
        print("❌ 모델 파일이 없습니다.")
        return
    
    clf, reg, scaler, features = pkg['classifier'], pkg['regressor'], pkg['scaler'], pkg['features']
    thresholds = pkg.get('thresholds', {'Male': {'prob_safe': 0.65, 'margin_safe': 7.0}, 'Female': {'prob_safe': 0.65, 'margin_safe': 10.0}})

    if not os.path.exists(DATA_FILE): return
    stats_db = build_team_stats(pd.read_csv(DATA_FILE))

    target_date = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(SCHEDULE_FILE):
        sch = pd.read_csv(SCHEDULE_FILE)
        
        # 🚨 [완전 통일] rename 로직 제거. schedule 파일은 이미 gdate, hname, aname임.
        sch['gdate'] = pd.to_datetime(sch['gdate'])
        
        today_games = sch[sch['gdate'] == target_date]
        if today_games.empty:
            future = sch[sch['gdate'] >= target_date].sort_values('gdate')
            if not future.empty:
                next_date = future.iloc[0]['gdate']
                print(f"📅 {target_date} 경기 없음 -> {next_date.strftime('%Y-%m-%d')} 경기 분석")
                today_games = future[future['gdate'] == next_date]
            else: return
    else: return

    print("-" * 60)
    print(f"🚀 AI 승부 예측")
    print("-" * 60)
    
    for _, game in today_games.iterrows():
        # 이제 그냥 hname, aname 쓰면 됨
        h = get_standardized_name(game['hname'])
        a = get_standardized_name(game['aname'])
        gender = get_gender(h)
        
        if h not in stats_db or a not in stats_db:
            print(f"⚠️ 데이터 부족: {h} vs {a}")
            continue
            
        st_h, st_a = stats_db[h], stats_db[a]
        
        input_data = pd.DataFrame([[
            st_h['elo'] - st_a['elo'],
            st_h['roll_attack_rate'] - st_a['roll_attack_rate'],
            st_h['roll_bs'] - st_a['roll_bs'],
            st_h['roll_ss'] - st_a['roll_ss'],
            st_h['roll_receive_rate'] - st_a['roll_receive_rate'],
            -(st_h['roll_err'] - st_a['roll_err'])
        ]], columns=features)
        
        X_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
        prob_home = clf.predict_proba(X_scaled)[0, 1]
        pred_diff = reg.predict(X_scaled)[0]
        
        p_win = prob_home if prob_home > 0.5 else 1 - prob_home
        winner = h if prob_home > 0.5 else a
        loser = a if prob_home > 0.5 else h
        abs_diff = abs(pred_diff)
        
        th_prob = thresholds[gender]['prob_safe']
        th_margin = thresholds[gender]['margin_safe']
        diff_att = st_h['roll_attack_rate'] - st_a['roll_attack_rate']
        att_adv = h if diff_att > 0 else a
        meta_warning = (winner != att_adv)
        
        est_score = ""
        guide_msg = []
        if p_win < th_prob:
            est_score = "3:2 (혼전/역배 주의)"
            guide_msg.append(f"👉 승패 : 🚫 패스 (AI 확신 {p_win*100:.1f}% < 기준 {th_prob*100:.0f}%)")
            guide_msg.append(f"👉 핸디캡 : 🍯 {loser} +1.5 플핸 (이변 가능성 높음)")
        else:
            if abs_diff >= th_margin:
                est_score = "3:0 or 3:1 (완승)"
                if meta_warning:
                     guide_msg.append(f"⚠️ [메타 경고] ELO 승자 vs 공격력 우위 불일치")
                     guide_msg.append(f"👉 {winner} 일반승 (마핸 위험)")
                else:
                    guide_msg.append(f"👉 {winner} -1.5 마핸 : 💎 강력 추천")
            else:
                est_score = "3:1 or 3:2 (접전승)"
                guide_msg.append(f"👉 {winner} 일반승 : ✅ 추천")
                guide_msg.append(f"👉 핸디캡 : {loser} +1.5 플핸")

        icon_g = "‍♂️ " if gender == 'Male' else "‍♀️ "
        print(f"🏐 [{icon_g}] {h} vs {a}")
        print(f"   📊 [전력] ELO {st_h['elo']:.0f} vs {st_a['elo']:.0f} | [메타] 공격차 {diff_att*100:+.1f}%p")
        print(f"   🏆 승자: {winner} ({p_win*100:.1f}%) | 기준점: {th_prob*100:.0f}%")
        print(f"   🔢 득실: {pred_diff:+.1f}점 | 마핸컷: {th_margin:.1f}점 -> {est_score}")
        for msg in guide_msg: print(f"      {msg}")
        print("-" * 60)

if __name__ == "__main__":
    predict_matchups()