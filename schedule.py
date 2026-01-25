import requests
import pandas as pd
import time
import os # 파일 유무 확인용
from datetime import datetime

# ==========================================
# 사용자 설정 영역
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "kovo_schedule_result.csv")

COLUMNS_TO_KEEP = [
    'gdate', 'gnum', 'seasonCode', 'leagueCode', 'round', 'gender',
    'hname', 'aname',
    'score',
    'hs1point', 'hs2point', 'hs3point', 'hs4point', 'hs5point',
    'as1point', 'as2point', 'as3point', 'as4point', 'as5point',
    'place', 'spectators', 'gstime', 'sptime', 'referee'
]

# ==========================================
# 핵심 함수
# ==========================================
def get_kovo_schedule(target_seasons=None):
    # 기본값 설정
    if target_seasons is None:
        target_seasons = ['018', '019', '020', '021', '022']
        
    print(f"🚀 일정 수집 시작: 대상 시즌 {target_seasons}")
    
    # -------------------------------------------------------
    # 1. API에서 최신 데이터 수집 (New Data)
    # -------------------------------------------------------
    new_games_list = []
    url = "https://user-api.kovo.co.kr/stat/game-schedule"
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.kovo.co.kr/'}

    for season in target_seasons:
        print(f"  📡 시즌 [{season}] 요청 중...", end=" ")
        try:
            params = {'seasonCode': season, 'leagueCode': '201', 'round': '', 'gcode': '001'}
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                game_list = data.get('payload', {}).get('content', [])
                if game_list:
                    print(f"✅ {len(game_list)}경기 확보")
                    new_games_list.extend(game_list)
                else:
                    print(f"⚠️ 데이터 없음")
            else:
                print(f"❌ 접속 실패 ({response.status_code})")
        except Exception as e:
            print(f"❌ 에러: {e}")
        time.sleep(0.5)

    if not new_games_list:
        print("❌ 수집된 신규 데이터가 없습니다. 종료합니다.")
        return

    # 신규 데이터프레임 생성 및 정리
    new_df = pd.DataFrame(new_games_list)
    valid_cols = [col for col in COLUMNS_TO_KEEP if col in new_df.columns]
    new_df = new_df[valid_cols]
    
    # -------------------------------------------------------
    # 2. 스마트 병합 (Merge Logic)
    # -------------------------------------------------------
    final_df = pd.DataFrame()

    if os.path.exists(OUTPUT_FILE):
        print("\n💾 기존 파일 발견! 데이터 병합 작업을 수행합니다.")
        try:
            # 기존 데이터 로드
            old_df = pd.read_csv(OUTPUT_FILE, dtype=str) # 안전하게 문자열로 로드
            
            # [핵심] 기존 데이터에서 '이번에 수집한 시즌들'을 삭제 (중복 방지)
            # 예: target_seasons=['022']라면, 기존 파일에서 '022' 데이터는 싹 지우고 새걸로 교체
            # seasonCode를 문자열로 확실하게 변환해서 비교
            cols_to_keep_mask = ~old_df['seasonCode'].astype(str).isin([str(s) for s in target_seasons])
            old_df_kept = old_df[cols_to_keep_mask]
            
            print(f"   - 기존 데이터: {len(old_df)}행")
            print(f"   - 유지할 과거 데이터: {len(old_df_kept)}행 (업데이트 대상 제외됨)")
            
            # 병합 (과거 데이터 + 신규 데이터)
            final_df = pd.concat([old_df_kept, new_df], ignore_index=True)
            print(f"   - 신규 추가 데이터: {len(new_df)}행")
            
        except Exception as e:
            print(f"⚠️ 기존 파일 읽기 실패 ({e}). 신규 데이터로만 덮어씁니다.")
            final_df = new_df
    else:
        print("\n✨ 기존 파일이 없습니다. 새 파일을 생성합니다.")
        final_df = new_df

    # -------------------------------------------------------
    # 3. 후처리 및 저장
    # -------------------------------------------------------
    # 날짜 정렬
    if 'gdate' in final_df.columns:
        final_df['gdate'] = pd.to_datetime(final_df['gdate'])
        sort_cols = ['gdate', 'gnum'] if 'gnum' in final_df.columns else ['gdate']
        final_df = final_df.sort_values(sort_cols)

    # 저장
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"🎉 스마트 업데이트 완료: {OUTPUT_FILE}")
    print(f"📂 최종 총 경기 수: {len(final_df)}경기")
    print("="*50)
    
    # 검증: 최근 경기 출력
    print("[데이터 정상 확인 (최근 3경기)]")
    today = pd.Timestamp.now().normalize()
    past_games = final_df[final_df['gdate'] <= today].tail(3)
    for _, row in past_games.iterrows():
        print(f" - {row['gdate'].strftime('%Y-%m-%d')} | {row['hname']} vs {row['aname']} | {row['score']}")

if __name__ == "__main__":
    # 테스트: 평소엔 전체 다 받다가, 업데이트할 땐 최근 것만 넣어도 됨
    # get_kovo_schedule() # 전체 실행
    get_kovo_schedule(target_seasons=['022']) # [테스트] 이번 시즌만 업데이트!