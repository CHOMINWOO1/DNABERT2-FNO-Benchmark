# DNABERT-2 adapter 및 LoRA 비교 계획

사용자가 승인한 7개 모델 × 5개 seed 비교다. 기본 설정은 `configs/comparative_study.json`, 실행 entry는 `python -m dnabert_fno.study`다. 첫 2-model PoC와 별도 output 폴더를 사용한다.

## 질문과 모델

| ID | 모델 | 학습 parameter 수 | 비교 목적 |
|---|---|---:|---|
| baseline | Frozen encoder + mean + linear | 1,538 | 선형 분류 기준 |
| mlp | Frozen + token-wise residual MLP, width 139 | 237,224 | 비선형 용량만으로 설명 가능한가 |
| no_spectral | FNO에서 spectral 경로만 제거, width 64 | 106,499 | 고정된 나머지 구조에서 spectral 기여 |
| cnn | Frozen + residual CNN, width 116, kernel 3 | 236,423 | 국소 token mixing과 비교 |
| fno | Frozen + residual FNO, width 64, modes 16 | 237,571 | Global spectral mixing |
| lora | 12개 fused QKV에 rank 6 LoRA + head | 222,722 | 기존 encoder 적응 방법과 비교 |
| lora_fno | 같은 LoRA + 같은 FNO + head | 458,755 | 두 방법의 결합 효과 |

실측 parameter 수는 각각 1,538 / 237,224 / 106,499 / 236,423 / 237,571 / 222,722 / 458,755다. MLP/CNN은 FNO의 1% 이내이며, LoRA는 약 6.3% 적다. No-spectral은 제거 ablation이므로 더 작다. LoRA+FNO는 더 크므로 단독 LoRA보다 좋아져도 용량 효과와 상보성을 분리할 수는 없다.

MLP는 FNO의 projection–pointwise–normalization–GELU–residual 형태를 유지하며 bottleneck을 넓혔다. CNN은 spectral 경로를 kernel 3 convolution으로 교체하고 원래 pointwise 경로도 유지한다. CNN의 boundary는 FNO와 맞춰 circular다. 모든 adapter는 실제 유효 token 길이에서 연산해 padding 영향을 차단한다. 이미 DNABERT-2 attention을 지난 hidden을 사용하므로 MLP/CNN도 원시 DNA의 비국소 정보를 간접적으로 포함할 수 있다.

LoRA는 `W + (alpha/r) B A`, rank=6, alpha=12, dropout=0.1이다. Q/K/V가 합쳐진 `Wqkv` 전체에 하나의 rank-6 업데이트를 적용하며 Q/V-only LoRA나 세 개의 독립 LoRA와 구분한다. A는 Kaiming, B는 zero init이다. 원본 encoder의 dropout은 eval로 고정하고 LoRA branch dropout만 학습 시 활성화한다. 원본 parameter는 고정하되 LoRA 학습의 autograd는 유지한다. 원본 parameter checksum을 학습 전후 비교한다.

## 데이터 통제

원본 GUE `prom_300_tata`의 6,130개 row를 합친 뒤 exact/reverse-complement 그룹을 만들었다. 동일 그룹에 상충하는 label이 있으면 전체 그룹을 제외하고, 일관된 그룹은 원래 방향의 서열 하나만 남겼다.

- 상충 label 그룹: 15개 제외
- 최종 고유 그룹: 6,036개
- 고정 split seed: 20260908
- Stratified 80/10/10: train 4,828 / dev 604 / test 604
- Split 사이 exact/RC 중복: 모두 0개
- 원래 row와 새 split의 대응: `data/GUE_clean/prom_300_tata/assignment.json`
- 출처, 제외 그룹, 전후 hash와 감사 결과: 같은 폴더의 `source.json`

새 split은 이미 조사한 같은 dataset을 재분할한 것이며 독립적인 외부 검증이 아니다. 근접 상동성, 염색체 수준 분리, pretraining 데이터 중복은 통제하지 않았다. 첫 PoC와 test가 달라 절대 점수의 전후 변화로 중복 제거 효과를 추정할 수 없다.

## 학습·선택·평가 순서

1. 프로토콜/config/code/environment/data hash를 output의 `protocol.json`에 고정한다.
2. 모든 모델에 같은 split과 seed 42,43,44,45,46을 사용한다. Head 초기값, 동일 epoch sample 순서는 seed별로 같다. LoRA 초기 A/B도 LoRA와 LoRA+FNO에서 동일하게 만든다.
3. Seed 42에서 방법별 두 LR을 dev로 비교한다. Frozen 방법은 `[3e-4,1e-3]`, LoRA 방법은 encoder LR `[1e-4,3e-4]`이며 head/FNO LR은 1e-3으로 고정한다. Dev best MCC 동률이면 후보 순서상 낮은 LR을 선택한다.
4. 선택한 LR은 나머지 4개 seed에 그대로 적용한다. 선택된 seed-42 trial을 재사용한다. 따라서 **최종 비교 35개 run + 탈락 LR pilot 7개 = 전체 42개 training trial**이다.
5. 모든 학습과 LR 선택이 끝난 뒤 `training_complete.json`을 기록한다. 그 뒤에만 정식 test metrics를 계산한다.
6. 작은 32-row engineering smoke run은 실행 흐름과 checkpoint/merge 기능 확인용으로 따로 수행했다. 그 점수는 LR나 구조 선택에 사용하지 않는다.

최대 25 epochs, early stopping patience 6, min_delta 1e-4, selection MCC. AdamW weight decay 1e-4, cosine LR, gradient clipping 1.0. Batch 8 × accumulation 4 = effective 32, 마지막 불완전한 accumulation은 실제 sample 수로 정규화한다. BF16 AMP와 float32/complex64 FFT를 사용한다. Encoder token cap은 512이고 실제 input은 300 bp다.

두 후보만의 작은 LR 탐색은 전역 최적화가 아니다. Seed 42가 LR 선택에 사용되므로 5-seed 결과와 함께 선택에 사용하지 않은 seed 43–46도 해석할 수 있다. Test threshold는 0.5로 고정한다.

## 비용과 보고

Frozen 방법은 공유 cache를 사용하고 LoRA는 매 batch online forward/backward를 수행한다. Cache 생성 시간과 peak를 별도 저장한다. 각 방법의 학습 시간은 selected trial의 실제 early-stopping 길이에 따른 비용이며, pilot 비용은 별도로 합산해야 한다.

주 test 점수는 학습된 unmerged 모델을 사용한다. LoRA 추론 처리량은 원본 weight에 업데이트를 병합한 모델로 측정한다. BF16에서 병합 전후 matmul 반올림이 달라질 수 있어 merged test metrics와 확률 차이, 변경된 예측 수도 저장한다. FP32에서 LoRA merge의 대수적 동등성은 unit test로 확인한다.

Inference는 동일한 test 앞 20개 batch, warmup 3회, batch별 synchronize, H2D 포함, tokenization/disk 제외다. PyTorch allocated/reserved peak와 전체 장치 VRAM을 구분한다. 타이밍은 단일 GPU 세션의 관측이며 반복 세션 불확실성은 별도로 측정하지 않는다.

MCC 주 지표와 Accuracy/F1/ROC-AUC/PR-AUC/average precision을 seed별로 저장한다. 평균과 sample SD(ddof=1), seed를 맞춘 MCC 차이와 양의 차이를 보인 seed 수를 보고한다. Seed SD는 optimizer/initialization 변동이며 dataset sampling uncertainty나 외부 일반화의 신뢰구간이 아니다. 여러 비교에 대한 유의확률을 무분별하게 해석하지 않는다.

## 실행과 재개

```powershell
# 최초 clean split 준비 (이미 준비되어 있으면 생략)
.\.venv\Scripts\python.exe scripts/prepare_clean_split.py

# 검증
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_lora.py

# 전체 study 실행 / 같은 명령으로 중단 후 재개
.\.venv\Scripts\python.exe -m dnabert_fno.study
```

`last.pt`에 epoch별 학습 parameter, optimizer/scheduler/scaler, RNG, 기록을 저장한다. 완료된 trial은 재학습하지 않는다. Protocol/code/data/environment hash가 바뀌면 새 output 디렉터리가 필요하다. 원본 DNABERT-2 가중치는 `.cache`에 고정 revision으로 보관한다.

결과 위치: `runs/comparative_study/summary.md`, `summary.json`, `all_runs.csv`. 각 seed의 checkpoint·dev 기록은 `trials/`, test predictions와 merged LoRA 수치는 `results/`에 있다. 실행 중 상태는 `status.json`에 기록한다.

참고: [LoRA 논문](https://arxiv.org/abs/2106.09685), [공식 LoRA 구현](https://github.com/microsoft/LoRA), [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2).
