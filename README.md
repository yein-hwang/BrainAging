# BrainAging
Reproducing Brain Aging paper using the PyTorch library.

### Paper
Investigation of Genetic Variants and Causal Biomarkers Associated with Brain Aging Jangho Kim, Junhyeong Lee, Seunggeun Lee medRxiv 2022.03.04.22271813; doi: https://doi.org/10.1101/2022.03.04.22271813

-----------------------------
## 2023년 7월 21일

**실험 내용:**
- 교차 검증(CV) 없이 진행중
- Batch size tuning
  - 다양한 배치 사이즈(8, 16, 32)로 실험 진행.
  - 배치 사이즈 8과 16에서는 큰 변화가 없었음.
  - 현재 배치 사이즈 32로 실험 중.
- Learning Rate tuning
  - `lr_scheduler.py` 파일에 구현된 `CustomCosineAnnealingWarmUpStart`
    - log: https://wandb.ai/yein-hwang/lr_test?workspace=user-yein-hwang 
  - 디폴트로 제공되는 `CosineAnealing`
- n_workers
  - `Dataloader` 과정에서 효율적인 병렬처리를 위해 설정할 수 있는 num_workers 조건 실험
  - `8`로 진행하는 것이 제일 효율적일 것 같음
  - 실험 결과 (전체적으로 1시간까지 돌려봄)
    - `> 64` : 한 시간 경과 후 training이 진행되지 않아 kill 한 후 아래 에러 확인  
      - `DataLoader worker (pid(s) 38532, 38911, 38974, 39606, 41309) exited unexpectedly`
    - `32` : 10% 진행 기준 23분 소요 -> 1시간 기준 27% 진행
    - `16` : 10% 진행 기준 29분 소요 -> 1시간 기준 19% 진행
    - `8` : 10% 진행 기준 16분 소요 -> 1시간 기준 39% 진행
    - `4` : 10% 진행 기준 26분 소요 -> 1시간 기준 30% 진행
- 추가 참고 사항
  - `main_lr.py` + `CNN_Trainer_lr.py` 파일로 실험을 진행 중.
  - 쉘 스크립트에서 `--ensemble_num` 옵션은 현재 모델 저장 폴더의 배치 사이즈를 구분하기 위해 사용 중.
 


-----------------------------
## 2023년 7월 22일

**실험 내용:**
- Learning Rate scheduling
  - 총 3가지 lr_scheduler 시도 중
    - step 단위 update: `CustomCosineAnnealingWarmUpStart`, `CosineAnealing`
      - `CustomCosineAnnealingWarmUpStart`
        - `learning_rate` 디렉토리에 위치
        - `lr_simulator.py`: Parameter setting에 따라 learning rate update 시각화 구현
        - ![image](https://github.com/yein-hwang/brain_aging/assets/109208473/c6dd356f-6b1b-4dc7-a2b7-fe2088d6f478)


    - epoch 단위: `ReduceLROnPlateau`
  - log: https://wandb.ai/yein-hwang/Brain_Aging
- `CNN_Trainer` Update
  - Epoch 3~ ) Load 후 loss 그대로 이어서 training 가능하도록 각 loss(train_mse, train_mse, valid_mse, valid_mae) 저장 추가
  - Epoch 6~ ) `CustomCosineAnnealingWarmUpStart` 경우, Load 후 초기 lr 셋팅 값이 너무 낮게 시작되는 문제 해결하기 위해 load 함수 수정
- `main` Update
  - lr scheduler 별 setting 재수정
 
-----------------------------
## 2023년 7월 25일

**실험 내용:**
- lr scheduler: `CustomCosineAnnealingWarmUpStart`으로 진행
- save setting 변경 (best model --> epoch 단위 모든 model) 후 14 epoch 부터 재시작
