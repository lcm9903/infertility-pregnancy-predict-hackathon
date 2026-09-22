# 난임 임신 성공 확률 예측 (두줄코딩조 제출파일)

기존 최종본을 바탕으로 피처 엔지니어링, fold별 인코딩, 멀티시드 앙상블, OOF 가중 블렌딩 등을 추가/개선한 최종 제출 노트북이다.

## 개선 사항 요약
- Funnel 파생 피처, 과거 실패 횟수 피처 추가
- 나이 × 동결 × 이식일 등 상호작용 피처 추가
- 기존 D 피처 세트와 개선 E 피처 세트 비교
- LightGBM/XGBoost 범주 인코딩을 **각 fold의 train에서만 fit**(리키지 방지), validation/test의 unseen category는 `-1` 처리
- Optuna 내부 교차검증에서도 fold별 인코딩 적용
- 최종 7-Fold에서 멀티시드(multi-seed) 앙상블
- OOF(Out-of-Fold) 기반 Logistic Stacking 추가
- 동일 OOF에서 가중치를 과도하게 탐색해 최고 점수를 최종 성능처럼 보고하던 절차 제거

> **중요한 구분**: A/B/C/D/E 중 최고 세트를 고르는 CV는 개발용 모델 선택 점수이며, 여러 후보 중 최고를 선택했기 때문에 완전히 독립적인 최종 성능 추정치는 아니다. 다만 피처 값 생성이나 fold 인코딩 과정에서 validation/test의 타깃 정보는 사용하지 않는다.

## 사용 데이터
- `train.csv`, `test.csv`, `sample_submission.csv` (Colab `/content`에 업로드 가정)
- 타겟: `임신 성공 여부`, 식별자: `ID`

## 실행 환경
- Google Colab 기준
- 주요 라이브러리: `pandas`, `numpy`, `matplotlib`, `koreanize-matplotlib`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna`, `shap`

## 진행 단계

### 1. 데이터 로드
- train/test/sample_submission 3개 CSV 로드, 파일 누락 시 즉시 에러로 안내
- 타겟 비율 확인

### 2. 최소 안전 전처리
- ID는 모델 입력에서 제외, train/test 공통 상수 컬럼 자동 제거
- 범주형 결측은 `'Missing'`으로, 숫자형 결측은 트리 모델이 직접 처리(임의 평균 대체로 구조적 결측 정보를 지우지 않음)

### 3. 파생변수 생성
- 배아 생성/저장 비율, 과거 시술 이력, 미세주입 비율, 동결 여부×이식 경과일 등 핵심 변수를 함수화

### 4. A/B/C/D/E 피처 세트 구성
| 세트 | 구성 |
|---|---|
| A | 원본만 |
| B | 원본 + 수치/비율/상태 파생 |
| C | 원본 + 상호작용/통합 범주 파생 |
| D | 원본 + 모든 파생 |
| E | 원본 + 모든 파생 v2 (추가 상호작용·수치 피처 포함) |

### 5. CatBoost 기반 피처 세트 비교
- 범주형을 자연스럽게 다루는 CatBoost로 동일 CV 조건에서 5개 피처 세트의 OOF ROC-AUC 비교
- 목적은 모델 경쟁이 아니라 **어떤 피처 세트가 실제로 성능을 높이는지 확인**하는 것
- 가장 높은 OOF AUC를 기록한 세트를 최종 입력(`BEST_FEATURE_SET`, `BEST_COLS`)으로 자동 선택

### 6. LightGBM/XGBoost용 범주형 인코딩
- 리키지 방지 ordinal encoder: 범주 매핑은 오직 현재 fold의 train에서만 생성, validation/test의 처음 보는 값은 `-1`

### 7. Optuna 하이퍼파라미터 탐색
- LightGBM, XGBoost, CatBoost 세 모델에 대해 3-Fold OOF AUC 기준으로 탐색
- 테스트 데이터와 최종 7-Fold 검증값은 튜닝 점수 계산에 사용하지 않음

### 8. 최종 7-Fold OOF + 테스트 예측
- 시드 `[42, 52, 62]` 멀티시드 앙상블로 seed별 OOF를 만든 뒤 평균

### 9. OOF 기반 Weighted Blending
- 단순 1:1:1 평균이 아니라 OOF에서 실제 AUC가 가장 높은 조합을 0.05 간격으로 탐색
- 이전 실험의 고정 가중치(0.30/0.10/0.60)도 별도 후보로 평가
- Logistic Regression 기반 OOF Stacking도 함께 비교

### 10. 최종 제출 파일 생성
- `FINAL_METHOD`(single_cb/equal/fixed/stack) 선택에 따라 최종 예측 산출
- `sample_submission`의 ID 순서와 `test.csv`의 ID 순서 일치 여부를 직접 대조한 뒤 반영
- 최종 산출물: `final_submission_high_performance.csv`

### 11. 타겟과의 연관성 분석
- 숫자형: 이진 타겟과 Spearman 상관(절댓값 상위 25개) — 비선형 관계도 포착 가능하도록 Pearson 대신 사용
- 범주형: 범주별 성공률 범위와 최소 표본수 함께 확인
- 이 분석은 설명 자료이며, 실제 피처 선택은 OOF AUC로 결정

### 12. SHAP 해석
- 최종 CatBoost 모델 기준 SHAP Summary Plot으로 예측 방향 해석

## 결과를 읽는 순서
1. 피처 세트 표에서 최고 OOF AUC와 Fold 표준편차 확인
2. 모델별 OOF AUC와 fold별 흔들림 비교
3. 단일 모델보다 앙상블 AUC가 실제로 높은지 확인
4. Spearman/범주별 성공률은 관계 탐색용, SHAP은 최종 모델의 예측 방향 해석용으로 사용
5. 최종 제출은 `final_submission_high_performance.csv` 한 개만 사용

## 참고
이 프로젝트는 팀 "두줄코딩조"의 난임 임신 성공 확률 예측 해커톤 제출 노트북이다.
