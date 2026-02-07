import pandas as pd
import os

# ==========================================
# 1. 설정
# ==========================================
INPUT_FILE = "kovo_schedule_result.csv"
OUTPUT_FILE = "kovo_analysis_ready.csv"

# 🔥 [최적화 완료] 적중률 1위 구간 (65.68%)
# 24-25 시즌 개막(2024-10-19) 직전인 10월 1일부터 사용
START_DATE = "2024-10-01" 

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
        if any(alias in name for alias in aliases):
            return std
    return name

def get_gender(team_name):
    men = ['대한항공', '현대캐피탈', 'KB손해보험', 'OK금융그룹', '한국전력', '우리카드', '삼성화재']
    return 'Male' if team_name in men else 'Female'

def process_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 파일 없음: {INPUT_FILE}")
        return

    print(f"📂 {INPUT_FILE} 로드 중...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. 날짜 처리
    df['gdate'] = pd.to_datetime(df['gdate'])
    
    # 2. 기간 필터링 (최적화된 2년치 데이터)
    df = df[df['gdate'] >= START_DATE].copy()
    
    # 3. 스코어 정제
    df = df.dropna(subset=['score'])
    df['score'] = df['score'].astype(str).str.replace(" ", "")
    df = df[df['score'].str.contains(":")] 

    # 4. 팀명 표준화
    df['h_std'] = df['hname'].apply(standardize_team_name)
    df['a_std'] = df['aname'].apply(standardize_team_name)
    
    # 5. 성별 구분
    df['gender'] = df['h_std'].apply(get_gender)

    # 6. 저장
    cols = ['gdate', 'gender', 'h_std', 'a_std', 'score']
    df[cols].sort_values('gdate').to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ 전처리 완료: {OUTPUT_FILE}")
    print(f"   - 기간: {START_DATE} ~ {df['gdate'].max().date()} (최근 2시즌)")
    print(f"   - 총 경기 수: {len(df)} 경기")

if __name__ == "__main__":
    process_data()