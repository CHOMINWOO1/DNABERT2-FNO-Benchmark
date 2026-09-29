# DNABERT-2 + FNO: 첫 PoC 결과

2026-09-08, Windows 11 / NVIDIA GeForce RTX 5060 Ti 8GB에서 구현과 실제 학습을 완료했다. 원 요청의 DNABERT는 사용자의 후속 지시에 따라 DNABERT-2로 변경했다.

**이번 단일 seed 실험에서는 frozen DNABERT-2 위에 FNO adapter를 추가했을 때 5개 test 지표가 모두 향상됐다.** 추론은 더 느려졌다. 이 결과만으로 향상 원인을 Fourier mixing으로 분리하거나 long-range 성능을 주장할 수는 없다.

## Test 비교

GUE `prom_300_tata`, train/dev/test = 4,904/613/613, seed 42. 원래 split을 유지했다. Dev MCC로 checkpoint를 선택한 뒤 test를 평가했다.

| 모델 | 학습 parameter 수 | Accuracy | F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Frozen DNABERT-2 + head | 1,538 | 0.6949 | 0.6825 | 0.3891 | 0.7391 | 0.6944 |
| Frozen DNABERT-2 + FNO + head | 237,571 | **0.7830** | **0.7719** | **0.5657** | **0.8538** | **0.8341** |

Accuracy는 **8.81 percentage points**, MCC는 **0.1766** 상승했다. PR-AUC는 사다리꼴 적분이며 average precision은 별도 JSON/CSV에 저장했다. 모든 지표는 동일한 613개 test row로 계산했다.

FNO는 width 64, modes 16, 1 layer다. LayerNorm과 768→64→768 projection, learnable residual `alpha`를 사용한다. 선택된 FNO checkpoint의 alpha는 0.1751이다. Baseline과 FNO의 초기 head parameter SHA-256은 동일하며 frozen encoder parameter 116,477,952개에는 gradient가 저장되지 않았다.

## 시간과 메모리

| 항목 | Baseline | FNO |
|---|---:|---:|
| 완료 epoch / 선택된 epoch | 23 / 17 | 10 / 4 |
| 학습만의 총 시간 | 31.86 s | 50.17 s |
| 학습 epoch 평균 | 1.39 s | 5.02 s |
| Dev 평가·checkpoint 저장 포함 학습 구간 | 34.99 s | 60.62 s |
| Cached training+dev peak allocated | 80.52 MiB | 92.40 MiB |
| Encoder+head 추론 peak allocated | 545.27 MiB | 546.17 MiB |
| Encoder+head 추론 처리량 | 477.42 samples/s | 297.57 samples/s |

Frozen hidden cache 생성에 **13.86초**가 추가로 소요됐고 두 모델이 공유했다. 학습 시간은 캐시를 이용한 FNO/head 학습 비용이다. Encoder를 매 epoch 다시 계산하는 full pipeline 학습 시간으로 읽으면 안 된다. FNO 첫 epoch에는 FFT plan 초기화 비용도 포함된다. 두 모델의 epoch 수가 다른 이유는 동일한 patience=6 early stopping 규칙이다.

추론은 batch 8, test의 앞 20개 batch(160개 sample), warmup 3회로 측정했다. Encoder와 classifier, H2D 전송 및 CUDA 동기화를 포함하고 tokenization과 디스크 읽기는 제외한다. FNO 처리량은 약 37.7% 낮았다. 이는 단일 세션의 timing이고 반복 세션 간 오차 범위는 측정하지 않았다.

메모리는 해당 PyTorch process의 allocated memory다. Windows 바탕화면과 다른 앱이 점유한 VRAM은 포함하지 않는다. Reserved memory와 cache 생성 peak는 원본 JSON에 별도로 기록했다.

## 데이터 출처와 중복 민감도

저자의 Google Drive 링크는 이 환경에서 TLS 오류로 실패했다. 대신 고정 revision의 [dnagpt/GUE](https://huggingface.co/datasets/dnagpt/GUE/tree/e0bc5eec01b46aca338e34ed4ab8ba97e096fe3e/prom/prom_300_tata)와 [leannmlindsey/GUE](https://huggingface.co/datasets/leannmlindsey/GUE/tree/240df51a2cd2c0fb1d88efffaf86e8d1e477f932/GUE/prom_300_tata)를 교차 검증했다. 세 split의 파일 hash가 모두 같고, 표본 수는 [DNABERT-2 논문](https://arxiv.org/abs/2306.15006)의 GUE 통계와 일치한다. 원본 Google Drive 아카이브와 직접 hash를 대조하지는 못했다.

train/dev 사이 14개, train/test 사이 6개 동일 서열이 있었다. Reverse complement를 포함해도 중복 개수는 같다. Dev/test 사이는 0개다. Train 내부 중복 row는 56개, dev 1개, test 2개다.

본 실험 전에 정한 보조 평가로, train/dev와 동일하거나 reverse-complement-identical한 test row 6개를 제외했다. 남은 607개 결과는 다음과 같다.

| 모델 | Accuracy | F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6952 | 0.6827 | 0.3898 | 0.7398 | 0.6984 |
| FNO | 0.7809 | 0.7703 | 0.5616 | 0.8524 | 0.8324 |

이 보조 평가에서도 개선은 유지됐다. 다만 train/dev 중복을 제거해 checkpoint 선택을 다시 수행한 것은 아니며 근접 상동성이나 pretraining genome과의 중복은 통제하지 않았다.

## 검증

- 17개 테스트 통과: frequency 선택, odd/short FFT, padding·batch 구성 불변성, alpha=0 baseline 보존, frozen gradient, FNO 역전파, FP16/BF16 CUDA, partial accumulation 정규화, sample 순서와 coverage, metric 및 중복 감사.
- 저장된 두 checkpoint를 각각 다시 로드하고, encoder를 매번 실행하는 online 경로로 **전체 test 613개를 재추론**했다. 저장된 cache 기반 예측과 허용 오차 내에서 일치하며 6개 metric 값은 모두 동일했다. 모델별 최대 확률 차이는 `verification.json`에 기록했다.
- 합성 DNA로 **batch 8 × 512 tokens × hidden 768**의 BF16 forward/backward/AdamW update를 통과했다. Peak allocated **880.45 MiB**, reserved **992 MiB**였다. 이는 메모리·연산 점검이며 생물학적 성능 실험이 아니다.
- 512-token update 전후 encoder 전체 parameter SHA-256이 동일함을 확인했다.

## 결론의 범위

최소 기술 성공 기준인 `MCC_FNO >= MCC_baseline - 0.01`을 만족했고, 이번 설정에서는 5개 지표가 모두 개선됐다. 따라서 **이 task와 seed에서 frozen DNABERT-2 representation에 작은 FNO adapter를 결합하는 구조는 동작하며 유망한 비교 결과를 보였다**고 판단한다.

현재 자료는 300 bp 서열이며 BPE 길이는 특수 토큰 포함 47–76이다. 512는 token cap이고 실제 downstream context가 아니다. FNO는 BPE token index 위에서 동작하므로 균일한 bp 격자 연산도 아니다. 원래 Transformer 전체를 실행한 뒤 FNO를 추가하므로 attention 비용은 유지된다.

다음 연구 판단에는 여러 seed와 추가 task, parameter 수를 맞춘 비spectral adapter control이 먼저 필요하다. 특히 FNO의 train loss는 계속 감소하지만 dev MCC는 4 epoch 이후 개선되지 않았으므로 early stopping이 중요했다. 이 첫 비교만으로 통계적 유의성, Fourier 고유 효과, long-range 활용 향상, long-context 효율을 주장하지 않는다.

## 파일

- 재현 방법과 구현 설명: [README](../README.md)
- 기본 설정: [promoter.json](../configs/promoter.json)
- 보관된 수치·출처·검증: [결과 snapshot](results_seed42/comparison.json), [데이터 감사](results_seed42/data_audit.json), [검증 기록](results_seed42/verification.json)
- 실제 run: `runs/promoter_seed42/` — checkpoint, epoch logs, test predictions, comparison 표
- Model module: `dnabert_fno/models.py`; training entry: `dnabert_fno/train.py`
