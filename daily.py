import time
import sys
import os

try:
    import schedule as step1
    # import collect as step2 (삭제됨)
    import process as step3
    import train as step4
    import predict_t as step5
   
except ImportError as e:
    print("🚨 [오류] 필수 파일이 누락되었습니다.")
    print(f"   에러 내용: {e}")
    sys.exit()

def run_pipeline():
    print("\n" + "="*70)
    print("🏐 KOVO AI 승부예측 - 원클릭 자동화 (Daily Auto)")
    print("="*70 + "\n")

    start_total = time.time()

    # --- Step 1: 일정 ---
    print("▶️ [1/4] 경기 일정 업데이트 (schedule.py)...")
    try:
        if hasattr(step1, 'get_kovo_schedule'):
            try: step1.get_kovo_schedule(['022']) 
            except TypeError: step1.get_kovo_schedule() 
        else: print("   ⚠️ 함수 확인 필요")
    except Exception as e:
        print(f"   ⚠️ 일정 에러 (기존 파일 사용): {e}")
    print("   ✅ 완료\n")
    time.sleep(0.5)

    # --- Step 2: 전처리 ---
    print("▶️ [2/4] 데이터 전처리 (process.py)...")
    try:
        if hasattr(step3, 'process_data'): step3.process_data()
        else:
            print("   ⚠️ 함수 확인 필요")
            return 
    except Exception as e:
        print(f"   ❌ 전처리 실패: {e}")
        return
    print("   ✅ 완료\n")
    time.sleep(0.5)

    # --- Step 3: 학습 ---
    print("▶️ [3/4] 구간 분석 및 학습 (train.py)...")
    try:
        if hasattr(step4, 'train_model'): step4.train_model()
        else:
            print("   ⚠️ 함수 확인 필요")
            return
    except Exception as e:
        print(f"   ❌ 학습 실패: {e}")
        return
    print("   ✅ 완료\n")
    time.sleep(2) 

    # --- Step 4: 예측 ---
    print("▶️ [4/4] 최종 승부 예측 (predict_t.py)...")
    try:
        if hasattr(step5, 'predict'): step5.predict()
        else: print("   ⚠️ 함수 확인 필요")
    except Exception as e:
        print(f"   ❌ 예측 실패: {e}")
        return
    
    print(f"\n⏱️ 소요 시간: {time.time() - start_total:.1f}초")

if __name__ == "__main__":
    run_pipeline()