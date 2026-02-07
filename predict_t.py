import pickle
import pandas as pd
import os
from datetime import datetime, timedelta

# ==========================================
# 1. 설정 및 경로
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_FILE = os.path.join(BASE_DIR, "kovo_schedule_result.csv")
MODEL_FILE_MALE = os.path.join(BASE_DIR, "elo_model_male.pkl")
MODEL_FILE_FEMALE = os.path.join(BASE_DIR, "elo_model_female.pkl")

MEN_TEAMS = ['대한항공', '현대캐피탈', 'KB손해보험', 'OK금융그룹', '한국전력', '우리카드', '삼성화재']

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def standardize_team_name(name):
    if pd.isna(name): return ""
    name = str(name).replace(" ", "").upper()
    mapping = {
        '대한항공': ['대한항공', '점보스', 'JUMBOS'],
        '현대캐피탈': ['현대캐피탈', '스카이워커스', 'SKYWALKERS'],
        'KB손해보험': ['KB손해보험', 'KB', 'KBSTARS', '케이비', 'LIG'],
        'OK금융그룹': ['OK금융', 'OK', 'OK저축은행', 'OKMAN', '읏맨'],
        '한국전력': ['한국전력', 'KEPCO', '빅스톰', 'VIXTORM'],
        '우리카드': ['우리카드', '위비', 'WON', 'WOORICARD'],
        '삼성화재': ['삼성화재', '블루팡스', 'BLUEFANGS'],
        '흥국생명': ['흥국생명', '핑크스파이더스', 'PINKSPIDERS'],
        '현대건설': ['현대건설', '힐스테이트', 'HILLSTATE'],
        '정관장': ['정관장', 'KGC', '인삼공사', 'REDSPARKS'],
        'IBK기업은행': ['IBK', '기업은행', '알토스', 'ALTOS'],
        'GS칼텍스': ['GS칼텍스', 'KIXX', 'GS'],
        '도로공사': ['도로공사', '하이패스', 'HIPASS', '한국도로공사'],
        '페퍼저축은행': ['페퍼', '페퍼저축은행', 'AI', 'PEPPERS']
    }
    for std, aliases in mapping.items():
        if any(alias in name for alias in aliases): return std
    return name

def get_matches(target_dates):
    matches = []
    if os.path.exists(SCHEDULE_FILE):
        try:
            df = pd.read_csv(SCHEDULE_FILE)
            df['gdate_dt'] = pd.to_datetime(df['gdate'])
            for date_str in target_dates:
                target_dt = pd.to_datetime(date_str)
                daily_matches = df[df['gdate_dt'] == target_dt]
                for _, row in daily_matches.iterrows():
                    h = standardize_team_name(row['hname'])
                    a = standardize_team_name(row['aname'])
                    gender = 'Male' if h in MEN_TEAMS else 'Female'
                    matches.append({'date': date_str, 'home': h, 'away': a, 'gender': gender})
        except: pass
    
    if not matches:
        backup = [
            {'date': '2026-02-08', 'home': '현대건설', 'away': '정관장', 'gender': 'Female'},
            {'date': '2026-02-08', 'home': '우리카드', 'away': '삼성화재', 'gender': 'Male'}
        ]
        matches = [m for m in backup if m['date'] in target_dates]
    return matches

# ==========================================
# 3. 메인 예측 함수
# ==========================================
def predict():
    if not os.path.exists(MODEL_FILE_MALE) or not os.path.exists(MODEL_FILE_FEMALE):
        print("❌ 모델 파일이 없습니다. train.py를 먼저 실행하세요.")
        return

    with open(MODEL_FILE_MALE, 'rb') as f: model_m = pickle.load(f)
    with open(MODEL_FILE_FEMALE, 'rb') as f: model_f = pickle.load(f)
    
    elo_m = model_m['elo']
    elo_f = model_f['elo']
    
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    matches = get_matches([today, tomorrow])
    
    print("\n" + "="*80)
    print(f"{'Date':^10} | {'Home':^10} vs {'Away':^10} | {'Diff':^8} | {'Pick':^10} | {'Grade':^10}")
    print("="*80)
    
    for m in matches:
        h, a, gen = m['home'], m['away'], m['gender']
        elo = elo_m if gen == 'Male' else elo_f
        
        h_score = elo.get(h, 1500)
        a_score = elo.get(a, 1500)
        
        if h_score > a_score: winner, diff = h, h_score - a_score
        else: winner, diff = a, a_score - h_score
            
        grade = "None"
        
        # 🔥 [최종 확정 로직] 누적 확률 기반 선형 등급
        if gen == 'Male':
            # 남자: 80(Gold) -> 140(Diamond)
            if diff >= 140:
                grade = "💎 Diamond"
            elif diff >= 80:
                grade = "🥇 GOLD"
            else:
                grade = "🥈 SILVER"
                
        else: # Female
            # 여자: 80(Gold) -> 180(Diamond)
            if diff >= 180:
                grade = "💎 Diamond"
            elif diff >= 80:
                grade = "🥇 GOLD"
            else:
                grade = "🥈 SILVER"

        print(f"{m['date']:^8} | {h:^8} vs {a:^8} | {diff:>6.1f} | {winner:^8} | {grade:^8}")

    print("="*80)
    # print("📌 [전략 가이드]")
    # print(" ♂️ 남자: 80점부터 'Gold', 140점 넘으면 'Diamond'")
    # print(" ♀️ 여자: 80점부터 'Gold', 180점 넘으면 'Diamond'")
    # print("="*80 + "\n")

if __name__ == "__main__":
    predict()