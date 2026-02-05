import pandas as pd
import json
import os

def process_kovo_data_final():
    print("Step 3: 데이터 파싱 및 분석용 파일 변환 (변수명 대통합)...")
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
    print("🚀 데이터 변환 작업 시작 (Standardized Columns 적용)...")
    
    for idx, row in df.iterrows():
        try:
            # 🚨 [핵심] 여기서부터 변수명을 schedule.py와 100% 일치시킵니다.
            meta = {
                'gdate': str(row.get('date', '')).split()[0],  # game_date -> gdate
                'seasonCode': row.get('season_code', ''),      # season -> seasonCode
                'round': row.get('round', ''),
                'gnum': row.get('gnum', ''),                   # game_num -> gnum
                'hname': row.get('home', ''),                  # home_team -> hname
                'aname': row.get('away', ''),                  # away_team -> aname
                'score': row.get('score', '')                  # set_score -> score
            }

            # JSON 파싱
            player_stats_str = row.get('player_stats', '[]')
            if isinstance(player_stats_str, str):
                try:
                    players = json.loads(player_stats_str)
                except:
                    players = []
            else:
                players = []

            # 선수별 데이터에 메타데이터 결합
            for p in players:
                # p 딕셔너리에 meta 딕셔너리를 합침
                # (주의: p에도 'tsname' 등이 있으므로 meta가 덮어쓰지 않도록 순서 주의)
                merged = {**meta, **p}
                all_players_rows.append(merged)
                
        except Exception as e:
            print(f"⚠️ Error at row {idx}: {e}")
            continue

    # 3. 데이터프레임 생성 및 후처리
    if all_players_rows:
        result_df = pd.DataFrame(all_players_rows)
        
        # 숫자 컬럼 강제 변환
        numeric_cols = [
            'point', 'attackSuccessRate', 'ats', 'att', 'bs', 'ss', 'rs', 'rt', 'err'
        ]
        existing_num_cols = [c for c in numeric_cols if c in result_df.columns]
        
        for col in existing_num_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        # 불필요 컬럼 제거
        drop_cols = ['profileImg', 'career', 'birthDate', 'teamCode'] 
        result_df = result_df.drop(columns=[c for c in drop_cols if c in result_df.columns], errors='ignore')

        # 컬럼 정렬 (표준 변수명 기준)
        cols = list(result_df.columns)
        priority = ['gdate', 'seasonCode', 'hname', 'aname', 'tsname', 'pname', 'position', 'point', 'score']
        sorted_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
        result_df = result_df[sorted_cols]

        # 저장
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"🎉 변환 완료: {output_file}")
        print(f"📊 총 데이터 행 수: {len(result_df)}")
        print(f"✅ 적용된 컬럼명: gdate, seasonCode, hname, aname, score 등 확인 완료.")
        
    else:
        print("⚠️ 변환할 데이터가 없습니다.")

if __name__ == "__main__":
    process_kovo_data_final()