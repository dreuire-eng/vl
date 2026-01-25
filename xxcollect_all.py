import cloudscraper
import pandas as pd
import time
import json
import sys
import os

# [필수] 시즌 코드 보정 함수
def get_correct_season_code(date_str):
    try:
        dt = pd.to_datetime(date_str)
        year = dt.year
        month = dt.month
        # 8월 기준 시즌 변경 (안전벨트)
        if (year == 2020 and month >= 8) or (year == 2021 and month < 8): return '017'
        elif (year == 2021 and month >= 8) or (year == 2022 and month < 8): return '018'
        elif (year == 2022 and month >= 8) or (year == 2023 and month < 8): return '019'
        elif (year == 2023 and month >= 8) or (year == 2024 and month < 8): return '020'
        elif (year == 2024 and month >= 8) or (year == 2025 and month < 8): return '021'
        elif (year == 2025 and month >= 8) or (year == 2026 and month < 8): return '022'
        return None
    except:
        return None

def collect_kovo_stats_final_optimized():
    print("Step 2: KOVO 데이터 수집 (최종 최적화: 초고속 + 중간저장)...")
    
    input_file = "kovo_schedule_result.csv"
    output_file = "kovo_player_stats_final.csv" # 저장할 파일명
    
    try:
        schedule_df = pd.read_csv(input_file, dtype=str)
        print(f"📂 '{input_file}' 로드 완료. 총 {len(schedule_df)}경기 대기 중.")
    except FileNotFoundError:
        print("❌ 일정 파일이 없습니다.")
        return

    # CloudScraper 생성
    scraper = cloudscraper.create_scraper()
    scraper.headers.update({
        'Referer': 'https://kovo.co.kr/',
        'Origin': 'https://kovo.co.kr',
        'x-service-name': 'user', 
        'accept': 'application/json'
    })
    
    collected_results = []
    total_games = len(schedule_df)
    
    print("\n🚀 데이터 수집 시작 (빠른 속도 주의!)")
    start_time = time.time()
    
    success_count = 0
    fail_count = 0

    for idx, row in schedule_df.iterrows():
        # 1. 파라미터 준비
        date_str = str(row['gdate'])
        
        # 시즌코드 보정
        correct_season = get_correct_season_code(date_str)
        s_code = correct_season if correct_season else str(row['seasonCode']).split('.')[0].zfill(3)
        l_code = str(row['leagueCode']).split('.')[0]
        
        # URL용 경기번호 (gnum에서 0 제거)
        raw_gnum = str(row['gnum']).split('.')[0].lstrip('0')
        
        # URL 구성
        url = f"https://user-api.kovo.co.kr/stat/game-schedule/{raw_gnum}"
        
        # 파라미터 (gcode 001 고정)
        params = {
            'seasonCode': s_code,
            'leagueCode': l_code,
            'gcode': '001'
        }
        
        try:
            response = scraper.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                payload = data.get('payload', {})
                
                if payload and ('player' in payload):
                    match_data = {
                        'date': row['gdate'],
                        'season_code': s_code,
                        'round': str(row['round']),
                        'gnum': raw_gnum,
                        'home': payload['game'].get('hname'),
                        'away': payload['game'].get('aname'),
                        'score': payload['game'].get('score'),
                        'game_meta': json.dumps(payload.get('game', {}), ensure_ascii=False),
                        'player_stats': json.dumps(payload.get('player', []), ensure_ascii=False),
                        'team_stats': json.dumps(payload.get('team', []), ensure_ascii=False)
                    }
                    collected_results.append(match_data)
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1

        except Exception as e:
            fail_count += 1
            print(f"❌ [에러] {e}")

        # [최적화 1] 딜레이 최소화 (0.05초)
        # 너무 빠르면 서버가 끊을 수 있으니 최소한의 예의만 갖춤
        time.sleep(0.05)
        
        # [최적화 2] 진행상황 출력 & 중간 저장 (50개마다)
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            # 중간 저장
            temp_df = pd.DataFrame(collected_results)
            temp_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"   [{idx + 1}/{total_games}] {(idx+1)/total_games*100:.1f}% 완료 | 성공: {success_count} | 💾 중간저장 완료")

    # 최종 저장
    print("\n💾 최종 데이터 저장 중...")
    result_df = pd.DataFrame(collected_results)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("="*50)
    print(f"🎉 수집 대장정 완료!")
    print(f" - 총 시도: {total_games}")
    print(f" - 성공: {success_count} ✅")
    print(f" - 실패: {fail_count}")
    print(f" - 파일: {output_file}")
    print("="*50)

if __name__ == "__main__":
    collect_kovo_stats_final_optimized()