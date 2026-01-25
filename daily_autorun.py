import time
import sys
import os

# seasons_to_collect = ['022']

# 각 단계별 파일(모듈) 불러오기
# (파일이 같은 폴더에 있어야 합니다)
try:
    import schedule as step1
    import collect_plus as step2
    import process as step3
    import train as step4
    import predict_t as step5
   
except ImportError as e:
    print("🚨 [오류] 파일을 찾을 수 없습니다!")
    print(f"   에러 내용: {e}")
    print("   👉 01~05번 코드 파일명이 'vk_01.py', 'vk_02.py'... 인지 확인해주세요.")
    sys.exit()

def run_pipeline():
    print("\n" + "="*60)
    print("🏐 KOVO AI 승부예측 - 원클릭 자동화 시스템 시작")
    print("="*60 + "\n")

    start_total = time.time()

    # --- Step 1: 일정 최신화 ---
    print("▶️ [1/5] 최신 경기 일정 업데이트 (vk_01.py)...")
    try:
        # 01번 코드의 메인 함수 실행
        # (함수 이름이 코드마다 다를 수 있으니, 아래 이름과 실제 파일 안의 함수명이 같은지 꼭 확인!)
        if hasattr(step1, 'get_kovo_schedule'):
            step1.get_kovo_schedule(['022']) # seasons_to_collect
        else:
            print("   ⚠️ 경고: vk_01.py 안에 실행 함수를 찾을 수 없습니다.")
    except Exception as e:
        print(f"   ❌ Step 1 실패: {e}")
        return # 여기서 중단

    print("   ✅ 일정 업데이트 완료.\n")
    time.sleep(1)


    # --- Step 2: 데이터 수집 ---
    print("▶️ [2/5] 경기 세부 데이터 크롤링 (vk_02.py)...")
    try:
        # 우리가 마지막으로 만든 스마트 업데이트 함수
        if hasattr(step2, 'collect_kovo_stats_smart_fix'):
            step2.collect_kovo_stats_smart_fix()
        elif hasattr(step2, 'collect_kovo_stats_final_safe'): # 혹시 이전 버전 이름일 경우
            step2.collect_kovo_stats_final_safe()
        else:
            print("   ⚠️ vk_02.py 실행 함수 확인 필요")
    except Exception as e:
        print(f"   ❌ Step 2 실패: {e}")
        return

    print("   ✅ 데이터 수집 완료.\n")
    time.sleep(1)


    # --- Step 3: 데이터 전처리 ---
    print("▶️ [3/5] 분석용 데이터 가공 (vk_03.py)...")
    try:
        if hasattr(step3, 'process_kovo_data_final'):
            step3.process_kovo_data_final()
        elif hasattr(step3, 'process_kovo_data'):
            step3.process_kovo_data()
    except Exception as e:
        print(f"   ❌ Step 3 실패: {e}")
        return

    print("   ✅ 데이터 가공 완료.\n")
    time.sleep(1)


    # --- Step 4: AI 모델 재학습 ---
    print("▶️ [4/5] AI 모델 최신화 및 학습 (vk_04.py)...")
    print("   (어제 경기 결과까지 반영하여 모델을 더 똑똑하게 만듭니다)")
    try:
        if hasattr(step4, 'train_logic_constrained_model_v2'):
            step4.train_logic_constrained_model_v2()
    except Exception as e:
        print(f"   ❌ Step 4 실패: {e}")
        return

    print("   ✅ 모델 학습 완료.\n")
    time.sleep(1)


    # --- Step 5: 오늘 경기 예측 ---
    print("▶️ [5/5] 오늘의 승부 예측 결과 (vk_05.py)...")
    print("="*60)
    try:
        if hasattr(step5, 'predict_matchups'):
            step5.predict_matchups()
    except Exception as e:
        print(f"   ❌ Step 5 실패: {e}")
        return

    print("="*60)
    end_total = time.time()
    print(f"🎉 모든 작업이 완료되었습니다! (소요시간: {end_total - start_total:.1f}초)")

if __name__ == "__main__":
    run_pipeline()