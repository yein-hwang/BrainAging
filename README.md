# brain_aging
Reproducing Brain Aging paper using the Tensorflow and PyTorch libarary.

# 실험 로그

## 2023년 7월 23일
-----------------------------
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
- 추가 참고 사항
  - `main_lr.py` + `CNN_Trainer_lr.py` 파일로 실험을 진행 중.
  - 쉘 스크립트에서 `--ensemble_num` 옵션은 현재 모델 저장 폴더의 배치 사이즈를 구분하기 위해 사용 중.
