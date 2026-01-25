import pandas as pd
import json
import os

def process_kovo_data_final():
    print("Step 3: 데이터 파싱 및 분석용 파일 변환 (Advanced)...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file  = os.path.join(BASE_DIR, "kovo_player_stats_final.csv")
    output_file = os.path.join(BASE_DIR, "kovo_analysis_ready.csv")

    # 1. 파일 로드
    try:
        df = pd.read_csv(input_file)
        print(f"📂 원본 데이터 로드: {len(df)}경기")
    except FileNotFoundError:
        print("❌ 파일이 없습니다. Step 2를 먼저 실행하세요.")
        return

    all_players_rows = []
    
    # 2. 파싱 및 펼치기 (Flatten)
    print("🚀 데이터 변환 작업 시작...")
    
    for idx, row in df.iterrows():
        try:
            # 경기 기본 메타데이터
            meta = {
                'game_date': str(row.get('date', '')).split()[0], # 시간 제거
                'season': row.get('season_code', ''),
                'round': row.get('round', ''),
                'game_num': row.get('gnum', ''),
                'home_team': row.get('home', ''),
                'away_team': row.get('away', ''),
                'set_score': row.get('score', '')
            }

            # JSON 파싱
            player_stats_str = row.get('player_stats', '[]')
            if pd.isna(player_stats_str) or player_stats_str == "": continue
            
            player_list = json.loads(player_stats_str)

            for p in player_list:
                # 메타데이터 복사 (Deep Copy 불필요, dict는 새로 생성)
                p_data = meta.copy()
                
                # API 데이터 병합
                # 팁: 약어(ats, ss 등)가 분석에 핵심이므로 그대로 둡니다.
                p_data.update(p)
                
                # [추가] 편의를 위한 파생 변수 생성
                # 예: 공격 효율 (성공 - 범실 - 차단) / 시도 -> 필요하면 여기서 계산 가능
                
                all_players_rows.append(p_data)

        except Exception as e:
            continue

    # 3. 데이터프레임 생성 및 후처리
    if all_players_rows:
        result_df = pd.DataFrame(all_players_rows)
        
        # [중요] 숫자 컬럼 강제 변환 (문자로 된 숫자들 처리)
        # 분석에 쓰일 주요 컬럼들이 숫자로 인식되게 함
        numeric_cols = [
            'point', 'attackSuccessRate', 'ats', 'att', 'bs', 'ss', 'rs', 'rt', 'err'
        ]
        # 실제 존재하는 컬럼만 골라서 변환
        existing_num_cols = [c for c in numeric_cols if c in result_df.columns]
        
        for col in existing_num_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        # [청소] 불필요한 컬럼 제거 (용량 최적화)
        drop_cols = ['profileImg', 'career', 'birthDate', 'teamCode'] # 예시
        result_df = result_df.drop(columns=[c for c in drop_cols if c in result_df.columns], errors='ignore')

        # 컬럼 정렬 (보기 좋게)
        cols = list(result_df.columns)
        priority = ['game_date', 'season', 'home_team', 'away_team', 'tsname', 'pname', 'position', 'point']
        sorted_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        result_df = result_df[sorted_cols]

        # 저장
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f"🎉 변환 완료! 분석 준비 끝.")
        print(f" - 총 데이터: {len(result_df)}행 (선수별 기록)")
        print(f" - 저장 파일: {output_file}")
        print("="*50)
        
    else:
        print("⚠️ 변환된 데이터가 없습니다.")

if __name__ == "__main__":
    process_kovo_data_final()