# BrainAging
Reproducing Brain Aging paper using the Tensorflow and PyTorch libarary.

### Paper
Investigation of Genetic Variants and Causal Biomarkers Associated with Brain Aging Jangho Kim, Junhyeong Lee, Seunggeun Lee medRxiv 2022.03.04.22271813; doi: https://doi.org/10.1101/2022.03.04.22271813

-----------------------------
## 2023년 7월 23일

**실험 내용:**
- 교차 검증(CV) 없이 진행중
- Batch size tuning
  - 다양한 배치 사이즈(8, 16, 32)로 실험 진행.
  - 배치 사이즈 8과 16에서는 큰 변화가 없었음.
  - 현재 배치 사이즈 32로 실험 중.
  - wandb에서 결과 확인 가능: https://wandb.ai/yein-hwang/lr_test?workspace=user-yein-hwang
- Learning Rate tuning
  - `lr_scheduler.py` 파일에 구현된 `CustomCosineAnnealingWarmUpStart` 사용 중.
  - 디폴트로 제공되는 CosineAnealing 실험 기록: https://wandb.ai/yein-hwang/BrainAging_lr_test?workspace=user-yein-hwang
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
