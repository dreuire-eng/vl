import cloudscraper
import pandas as pd
import time
import json
import os
import re
import shutil
from datetime import datetime

# ==========================================
# 1. 유틸리티 함수
# ==========================================
def get_correct_season_code(date_str):
    try:
        dt = pd.to_datetime(date_str)
        year = dt.year
        month = dt.month
        if (year == 2020 and month >= 8) or (year == 2021 and month < 8): return '017'
        elif (year == 2021 and month >= 8) or (year == 2022 and month < 8): return '018'
        elif (year == 2022 and month >= 8) or (year == 2023 and month < 8): return '019'
        elif (year == 2023 and month >= 8) or (year == 2024 and month < 8): return '020'
        elif (year == 2024 and month >= 8) or (year == 2025 and month < 8): return '021'
        elif (year == 2025 and month >= 8) or (year == 2026 and month < 8): return '022'
        return None
    except:
        return None

def is_game_finished(score_str):
    """ 스코어에 숫자가 포함되어 있고 0:0이 아니면 종료된 것으로 판단 """
    if pd.isna(score_str): return False
    score_str = str(score_str).strip()
    if score_str in ["", "0:0", "0 : 0", "0:0(0:0)", "0 : 0 (0 : 0)"]: return False
    if re.search(r'[1-3]', score_str): return True
    return False

# ==========================================
# 2. 메인 수집 함수 (Logic fix applied)
# ==========================================
def collect_kovo_stats_smart_fix():
    print("Step 2: KOVO 상세 데이터 수집 (Logic Fixed)...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 저장할 파일들 경로 수정
    schedule_file = os.path.join(BASE_DIR, "kovo_schedule_result.csv")
    output_file   = os.path.join(BASE_DIR, "kovo_player_stats_final.csv")
    backup_file   = os.path.join(BASE_DIR, "kovo_player_stats_final.bak")
    
    # 1. 일정 파일 로드
    try:
        schedule_df = pd.read_csv(schedule_file, dtype=str)
        # 날짜 형식 변환
        schedule_df['gdate_dt'] = pd.to_datetime(schedule_df['gdate'])
        print(f"📂 일정 파일 로드: {len(schedule_df)}경기")
    except FileNotFoundError:
        print("❌ 일정 파일(01번)이 없습니다.")
        return

    # 2. 기존 데이터 로드 & 상태 매핑
    collected_status = {} # {'gnum': 'score'}
    existing_data = []

    if os.path.exists(output_file):
        try:
            shutil.copy(output_file, backup_file)
            print(f"🛡️ 백업 완료: {backup_file}")
            
            existing_df = pd.read_csv(output_file, dtype=str)
            existing_data = existing_df.to_dict('records')
            
            for _, row in existing_df.iterrows():
                # gnum을 확실하게 정제 (0 제거)
                clean_gnum = str(row['gnum']).split('.')[0].lstrip('0')
                collected_status[clean_gnum] = str(row['score'])
                
            print(f"💾 기존 데이터: {len(collected_status)}경기 확인됨.")
        except Exception as e:
            print(f"⚠️ 기존 파일 읽기 실패: {e}")

    # 3. 업데이트 대상 정밀 선별
    tasks_to_do = []
    today_dt = pd.Timestamp.now().normalize()
    
    print("\n🔍 업데이트 대상 분석 중...")
    
    for _, row in schedule_df.iterrows():
        raw_gnum = str(row['gnum']).split('.')[0].lstrip('0')
        sched_score = str(row['score'])
        game_date = row['gdate_dt']
        
        # [핵심 로직 수정]
        # 1. 미래의 경기는 무조건 패스
        if game_date > today_dt:
            continue
            
        # 2. 과거~오늘 경기인데 데이터 파일에 아예 없다? -> 수집 대상
        if raw_gnum not in collected_status:
            # 단, CSV상 0:0이라도 날짜가 지났으면 혹시 모르니 수집 시도 (API는 업데이트 됐을 수 있음)
            tasks_to_do.append(row)
            continue
            
        # 3. 데이터 파일에 있는데, 저장된 스코어가 0:0 (미완성)이다? -> 업데이트 대상
        saved_score = collected_status[raw_gnum]
        if not is_game_finished(saved_score):
            # 날짜가 지났거나 오늘이면 다시 긁어봄
            if game_date <= today_dt:
                tasks_to_do.append(row)

    # 중복 제거 (혹시 모를)
    # tasks_to_do는 DataFrame Row의 리스트임
    
    total_tasks = len(tasks_to_do)
    if total_tasks == 0:
        print("✅ 모든 과거 경기가 업데이트 되어 있습니다.")
        return

    print(f"🚀 {total_tasks}경기의 데이터를 확인/수집합니다.")
    
    # 4. 크롤링 수행
    scraper = cloudscraper.create_scraper()
    new_data_buffer = []
    
    # 업데이트할 gnum 목록 추출
    update_gnums = [str(t['gnum']).split('.')[0].lstrip('0') for t in tasks_to_do]
    
    # 기존 데이터에서 이번에 업데이트할 놈들은 미리 제거 (덮어쓰기 준비)
    final_existing_data = [d for d in existing_data if str(d['gnum']).split('.')[0].lstrip('0') not in update_gnums]

    for idx, row in enumerate(tasks_to_do):
        raw_gnum = str(row['gnum']).split('.')[0].lstrip('0')
        date_str = str(row['gdate']).split()[0]
        
        # 시즌 코드 계산
        correct_season = get_correct_season_code(date_str)
        s_code = correct_season if correct_season else str(row['seasonCode']).split('.')[0].zfill(3)
        l_code = str(row['leagueCode']).split('.')[0]

        url = f"https://user-api.kovo.co.kr/stat/game-schedule/{raw_gnum}"
        params = {'seasonCode': s_code, 'leagueCode': l_code, 'gcode': '001'}
        
        try:
            response = scraper.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                payload = data.get('payload', {})
                player_list = payload.get('player', [])
                
                # [중요] API가 실제로 유효한 데이터를 줬는지 확인
                # 경기가 안 끝났으면 player_list가 비어있거나 score가 0:0일 것임
                api_score = payload.get('game', {}).get('score', '0:0')
                
                if payload and player_list and is_game_finished(api_score):
                    match_data = {
                        'date': date_str,
                        'season_code': s_code,
                        'round': str(row['round']),
                        'gnum': raw_gnum,
                        'home': payload['game'].get('hname'),
                        'away': payload['game'].get('aname'),
                        'score': api_score, # API에서 받은 최신 스코어 사용
                        'game_meta': json.dumps(payload.get('game', {}), ensure_ascii=False),
                        'player_stats': json.dumps(player_list, ensure_ascii=False),
                        'team_stats': json.dumps(payload.get('team', []), ensure_ascii=False)
                    }
                    new_data_buffer.append(match_data)
                    print(f"   [{idx+1}/{total_tasks}] {date_str} {match_data['home']} vs {match_data['away']} ({api_score}) ✅ 업데이트")
                else:
                    # 경기가 취소됐거나 아직 시작 안 함
                    print(f"   [{idx+1}/{total_tasks}] {date_str} (아직 결과 없음/0:0) 💤 Skip")
            else:
                print(f"   [Error] Status: {response.status_code}")
        except Exception as e:
            print(f"   [Exception] {e}")

        time.sleep(0.05)

        # 중간 저장
        if len(new_data_buffer) > 0 and len(new_data_buffer) % 10 == 0:
            temp_df = pd.DataFrame(final_existing_data + new_data_buffer)
            temp_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 5. 최종 병합 및 저장
    # (새로 수집된 게 없어도, existing_data가 변경되었을 수 있으므로 - 중복제거 등 - 저장 루틴 실행)
    final_df = pd.DataFrame(final_existing_data + new_data_buffer)
    
    if 'date' in final_df.columns:
        final_df['date'] = pd.to_datetime(final_df['date'])
        final_df = final_df.sort_values(['date', 'gnum'])
        
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"🎉 동기화 완료!")
    print(f" - 총 데이터: {len(final_df)}경기")
    print(f" - 이번에 업데이트됨: {len(new_data_buffer)}경기")
    if len(new_data_buffer) == 0:
        print(" (팁: 만약 어제 경기가 안 들어왔다면, 01번 코드를 먼저 실행해서 일정표를 갱신해보세요)")
    print("="*50)

if __name__ == "__main__":
    collect_kovo_stats_smart_fix()