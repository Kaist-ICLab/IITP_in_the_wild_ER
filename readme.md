# 코드 관련 폴더 구조 설명

## data
- 원본 데이터는 모두 `23EmoWorkerField`에 있는데, 프라이버시 이슈로 공개를 제한합니다.
- `data` 폴더에는 최종적으로 학습에 사용된 데이터가 정리되어 있습니다. 
    - 이 데이터는 `23EmoWorkerField`의 `3_feature_extraction`에 있는 `In_the_wild_ER`과 동일한 파일입니다.

## results
- `results` 폴더는 논문에 최종적으로 포함된 결과들을 정리한 폴더입니다.
- `Data_ablation_RF`, `ER_ablation`, `general`, `hybrid`, `personalization` 등의 서브 폴더가 존재하며, 각각 다른 실험 분석 결과를 저장하고 있습니다.


## src
- `src` 폴더에는 데이터 전처리, 피처 추출, 모델링과 관련된 모든 코드가 포함되어 있습니다.
- `data preprocessing`
  - `cleaning`: 데이터 정제 과정 관련 코드
  - `feature_extraction`: 피처 추출 관련 코드
- `modeling`
  - `metric.py`: 모델 평가 지표 관련 코드
  - `MLmodel.py`: 실험에 사용된 머신러닝 모델들에 관련된 코드
  - `preprocessing.py`: 모델링에 필요한
   전처리 관련 코드

## experiment
- `experiment.ipynb` 파일은 논문에 포함된 실험을 수행한 코드입니다.


### 그외의 주의사항 :  환경 변수 설정 관련 사항
- `envs`에 있는 data path는 반드시 `'23EmoWorkerFeild'`폴더에 관련된 경로여야 합니다.
- 모든 코드에서 사용되는 `envs` 파일 경로를 실제 `envs`가 위치한 경로로 변경해야 합니다.
