# 긴 context 탐색 실험: GM12878 EPI

질문은 같은 enhancer–promoter 쌍에서 context가 늘어날 때 frozen FNO와 CNN/LoRA의 성능 차이가 어떻게 변하는가이다. 실험은 길이 1,000 / 2,000 / 5,000bp × CNN/FNO/LoRA × seed 42/43/44, 총 27개 run이다. 이번에는 전체 GUE+ 재현이 아닌 고정 subset 탐색 실험을 수행한다.

## 데이터

출처는 [GUE+를 기술한 DNABERT-2 논문](https://arxiv.org/html/2306.15006v2)과 [공개 GUE_v2 미러](https://huggingface.co/datasets/genomic-benchmarks/GUE_v2)다. 미러 revision은 `f1290ccd49d4c7f8014cac47c93779aed06a294b`이다. GM12878 train/val/test 10,000/2,000/2,000개를 다운로드했고, 각 compressed file의 SHA256이 이 revision의 LFS hash와 일치함을 확인했다. 공식 Google Drive 아카이브는 TLS 연결 실패로 byte 단위 대조하지 못했다.

원본은 enhancer 3,000bp와 promoter 2,000bp의 두 컬럼을 가진다. 두 구간 사이의 실제 유전체 DNA가 들어 있는 연속 5kb 서열은 아니다. 입력은 enhancer 다음 promoter를 이어 붙인다. 짧은 context는 enhancer와 promoter에서 각각 중심 구간을 3:2 비율로 취한다.

| 총 bp | Enhancer 중심 | Promoter 중심 |
|---|---:|---:|
| 1,000 | 600 | 400 |
| 2,000 | 1,200 | 800 |
| 5,000 | 3,000 | 2,000 |

전체 raw split을 합친 뒤 pair exact/RC 중복 및 상충 label을 검사한다. Enhancer와 promoter 각각에서 모든 길이의 cropped view가 exact/RC로 같으면 하나의 component 그룹으로 묶는다. Component 그룹을 label과 무관하게 60/20/20으로 독립 배정하고, enhancer와 promoter의 배정 split이 같은 pair만 남긴다. 그 안에서 train 1,024 / dev 256 / test 256개를 label별 동수로 고정 추출한다. Split seed는 20260909다.

세 길이 모두 같은 source pair, label, row 순서를 사용한다. 각 길이에서 enhancer·promoter·연결 서열의 split 간 exact/RC 겹침은 모두 0이다. Near-homology, shifted genomic overlap, chromosome 분리는 확인하지 못했다. 공식 split과 달라 공식 논문의 점수와 직접 비교하지 않는다. 원본 매핑과 제외/중복 감사는 `data/long_epi_pilot/source.json`에 저장한다.

## 학습

- Backbone: DNABERT-2-117M, revision `7bce263b15377fc15361f52cfab88f8b586abda0`.
- 이전 실험과 같은 CNN width 116, FNO width 64/modes 16/1 block, fused-QKV LoRA rank 6/alpha 12. 학습 parameter는 각각 236,423 / 237,571 / 222,722다.
- LR 0.0003 고정. LoRA head LR은 0.001. 이전 promoter 실험에서 선택한 설정을 사용하며 이 task에서 새 LR 탐색을 하지 않는다.
- 최대 12 epochs, patience 4, dev MCC checkpoint 선택, min_delta 0.0001. Effective batch 32 = batch 2 × accumulation 16.
- 동일한 sample 순서를 길이 사이에도 유지하기 위해 length bucketing을 끈다. 같은 seed의 head/adapter/LoRA 초기값을 유지한다.
- BF16, float32 FFT, 원본 encoder 가중치 고정. Frozen CNN/FNO는 공유 feature cache, LoRA는 online forward/backward.
- 상한 `max_tokens=5002`로 모든 실제 입력을 보존한다. 기존 512-token PoC runner는 수정하지 않고, 동적 ALiBi를 사용하는 별도 runner에서 처리한다.
- 모든 27개 학습이 끝난 뒤에만 정식 test metrics를 계산한다. 길이별 MCC/Accuracy/F1/ROC-AUC/PR-AUC, paired MCC 차이, 학습 및 추론 시간과 메모리를 저장한다.
- 짧은 run과 작은 subset 및 제한된 LR는 exploratory 조건이다. 수렴이나 최적 성능을 보장하지 않으며 seed SD는 표본 불확실성의 신뢰구간이 아니다.

Test 평가 전 추가로 정의한 불확실성 분석은 paired stratified bootstrap 5,000회(seed 20260910)다. Test label별 같은 row 표본을 모든 모델·길이·학습 seed에 함께 적용하고, 3개 고정 학습 모델의 평균 MCC 차이에 대한 percentile 95% 구간을 구한다. 길이별 FNO−CNN/LoRA, 각 모델의 5kb−1kb 변화, FNO−LoRA 차이의 5kb−1kb 변화를 보고한다. 학습 모델과 이번 test cohort에 조건부인 기술적 구간이며, 같은 enhancer/promoter를 공유하는 pair 사이의 상관, 재학습·염색체 표본 불확실성·다중 비교 보정은 포함하지 않는다.

## 사전 실행 확인

실제 데이터의 BPE token 범위(special token 포함)는 1kb에서 167–233, 2kb에서 358–452, 5kb에서 902–1,074였다. 잘린 입력은 0개다. `runs/long_context_probe/token_audit.json`에 split별 값을 저장했다.

합성 random DNA를 사용하는 LoRA+FNO capacity probe에서 batch 1의 5kb(1,067 tokens), 10kb(2,133 tokens)가 모두 3개 optimizer step을 통과했다. Peak allocated는 각각 2,209.8 / 5,929.2 MiB였다. 5kb batch 2(최대 1,068 tokens)도 통과했고 peak는 3,644.6 MiB였다. 이 probe는 실행 가능성만 확인하며 생물학적 정확도나 10kb task 성능을 측정하지 않았다.

## 실행

처음 데이터를 준비하는 환경에서는 다음 두 단계를 먼저 실행한다. 이미 준비된 데이터는 다시 만들지 않는다. 다운로드 스크립트는 고정 revision과 SHA256을 검사하며, `--check-only` 옵션은 네트워크나 파일 변경 없이 기존 파일만 검사한다.

```powershell
.\.venv\Scripts\python.exe scripts/download_long_epi.py
.\.venv\Scripts\python.exe scripts/prepare_long_epi.py
```

학습 실행 또는 동일 조건에서 재개:

```powershell
.\.venv\Scripts\python.exe scripts/run_long_epi.py
```

중단 후 같은 명령으로 재개한다. Config/source/data/environment identity가 바뀌면 새 output 디렉터리가 필요하다. 진행 상황은 `runs/long_epi_pilot/status.json`과 해당 `bp*/status.json`에 기록된다. 이전 300bp 실험의 코드와 결과는 유지한다.

완료 후 저장된 예측 검증과 보고용 결과 생성:

```powershell
.\.venv\Scripts\python.exe scripts/verify_long_epi.py
.\.venv\Scripts\python.exe scripts/bootstrap_long_epi.py
.\.venv\Scripts\python.exe scripts/export_long_epi.py
```
