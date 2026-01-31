import pandas as pd
import numpy as np
import pickle
import os

# =========================================================
# 1. 설정 및 모델 로드
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "kovo_dual_model.pkl")

def inspect_model():
    print("🚀 AI 모델 내부 가중치(Weights) 해부")
    print("-" * 60)

    if not os.path.exists(MODEL_FILE):
        print("❌ 모델 파일이 없습니다. 04번을 먼저 실행하세요.")
        return

    with open(MODEL_FILE, "rb") as f:
        pkg = pickle.load(f)
    
    # 모델 요소 추출
    reg = pkg['regressor']   # 득실마진 예측기 (Ridge)
    clf = pkg['classifier']  # 승패 예측기 (LogisticRegression)
    features = pkg['features']
    
    print("✅ 모델 로드 성공!")
    print(f"   - 사용된 피처: {features}")
    print("-" * 60)

    # =========================================================
    # 🔍 1. [득실마진] 점수에 영향을 주는 요소 (Regressor)
    # =========================================================
    print("\n📊 1. [득실마진] 점수차를 벌리는 핵심 요인은? (Ridge Model)")
    print("   (가중치가 클수록 점수차에 결정적인 영향을 줌)")
    print("-" * 60)
    
    # 가중치 추출
    reg_coefs = reg.coef_
    
    # 데이터프레임으로 정리
    df_reg = pd.DataFrame({
        'Feature': features,
        'Weight': reg_coefs,
        'Abs_Weight': np.abs(reg_coefs) # 중요도 순 정렬용
    })
    
    # 중요도 순 정렬
    df_reg = df_reg.sort_values('Abs_Weight', ascending=False)
    
    for _, row in df_reg.iterrows():
        name = row['Feature']
        weight = row['Weight']
        
        # 해석
        impact = "🟢 점수 벌림 (유리)" if weight > 0 else "🔴 점수 까먹음 (불리)"
        bar = "█" * int(abs(weight) * 2) # 시각화
        
        print(f"{name:<15} | {weight:>8.4f} | {impact} {bar}")

    print("\n   💡 [해석 팁]")
    print("      - diff_att (공격성공률)가 높으면 점수를 팍팍 냅니다.")
    print("      - diff_fault (범실)는 부호를 반전했으므로, 양수면 '범실이 적어서 좋다'는 뜻입니다.")
    print("      - ELO는 '기본 체급'이라 베이스 점수를 깔고 갑니다.")

    # =========================================================
    # 🔍 2. [승률] 승패를 가르는 결정적 한방 (Classifier)
    # =========================================================
    print("\n\n📊 2. [승패확률] 이기는 팀의 조건은? (Logistic Regression)")
    print("   (이 값이 높을수록 승리 확률을 높게 평가함)")
    print("-" * 60)
    
    clf_coefs = clf.coef_[0] # 로지스틱은 2차원 배열이라 [0] 인덱싱 필요
    
    df_clf = pd.DataFrame({
        'Feature': features,
        'Weight': clf_coefs,
        'Abs_Weight': np.abs(clf_coefs)
    })
    df_clf = df_clf.sort_values('Abs_Weight', ascending=False)
    
    for _, row in df_clf.iterrows():
        name = row['Feature']
        weight = row['Weight']
        impact = "🔥 승률 UP" if weight > 0 else "❄️ 승률 DOWN"
        bar = "█" * int(abs(weight) * 2)
        
        print(f"{name:<15} | {weight:>8.4f} | {impact} {bar}")

if __name__ == "__main__":
    inspect_model()