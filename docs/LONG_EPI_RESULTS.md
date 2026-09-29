# DNABERT-2 긴 입력 비교 결과: GM12878 EPI

**1kb·2kb·5kb × CNN/FNO/LoRA × 3개 seed, 총 27회 학습과 test 평가를 완료했다. 이번 조건에서는 긴 입력이 FNO의 상대적 성능을 높인다는 근거를 얻지 못했다.** FNO의 평균 MCC는 1kb 0.2289에서 5kb 0.1416으로 감소했다. 5kb에서는 LoRA의 평균이 높았지만, 표본과 seed 불확실성이 커서 일반적인 우열을 확정할 결과는 아니다.

RTX 5060 Ti 8GB에서 BF16으로 실행했다. 학습·표현 추출·test 평가 전체 runner의 실행 시간은 2026-09-08 16:48:09–17:57:54 KST, 약 69.75분이다. 데이터 준비, 사전 capacity probe 및 사후 검증 시간은 별도다.

## 같은 데이터에서 길이만 바꾼 비교

GM12878 enhancer–promoter interaction 분류에서 train 1,024 / dev 256 / test 256쌍을 고정했고, 각 split의 두 label 수는 같다. 세 길이에 동일한 원본 쌍·label·순서를 사용했다. Enhancer와 promoter 각각의 중심 구간을 3:2 비율로 잘라 이어 붙였다.

| 입력 길이 | Enhancer / promoter bp | Frozen + CNN MCC | Frozen + FNO MCC | LoRA MCC |
|---:|---:|---:|---:|---:|
| 1kb | 600 / 400 | **0.2411 ± 0.0101** | 0.2289 ± 0.0119 | 0.2087 ± 0.0127 |
| 2kb | 1,200 / 800 | 0.1705 ± 0.0121 | 0.1696 ± 0.0669 | **0.1907 ± 0.1031** |
| 5kb | 3,000 / 2,000 | 0.1356 ± 0.0650 | 0.1416 ± 0.0520 | **0.2171 ± 0.0817** |

수치는 seed 42/43/44의 평균 ± sample SD다. 굵은 숫자는 해당 행의 가장 높은 평균이며 통계적으로 확정된 승자를 뜻하지 않는다. 모든 checkpoint와 학습률은 test 평가 전에 고정했다. 전체 27개 학습이 끝난 뒤 주 test 지표를 한 번 평가했으며, LoRA는 별도로 병합 후 지표도 측정했다.

**5kb는 두 구간을 연결한 입력 길이이며, 사이의 실제 유전체 서열이 포함된 연속 5kb 구간은 아니다.** 각 길이에서 별도로 학습했으므로 짧은 입력에서 학습한 모델의 긴 입력 전이 성능도 측정하지 않았다. BPE token 범위는 1kb 167–233, 2kb 358–452, 5kb 902–1,074이며 잘린 입력은 0개다. 10kb는 합성 DNA로 실행 가능성만 확인했고, 생물학적 정확도 실험에는 포함하지 않았다.

학습 parameter 수는 CNN 236,423 / FNO 237,571 / LoRA 222,722다. 같은 effective batch 32를 사용했으며 최대 12 epochs, patience 4, dev MCC로 checkpoint를 선택했다. LR은 이전 promoter 실험에서 정한 값을 사용했고 이번 task에서 새 탐색은 하지 않았다. 세 방법의 계산량이나 학습 시간까지 같게 맞춘 비교는 아니다. 자세한 조건은 [프로토콜](LONG_EPI_PROTOCOL.md)에 있다.

![길이별 test MCC, 학습 시간, GPU 할당 메모리](C:/Users/public-user/Documents/ChatGPT/FNO/docs/long_epi_results/length_comparison.png)

## 성능 변화의 해석

| 1kb → 5kb 변화 | 평균 ΔMCC ± paired seed SD | MCC가 오른 seed |
|---|---:|---:|
| CNN | −0.1055 ± 0.0690 | 0/3 |
| FNO | −0.0873 ± 0.0618 | 0/3 |
| LoRA | +0.0084 ± 0.0887 | 2/3 |

FNO는 3개 seed 모두에서 5kb MCC가 1kb보다 낮았다. 5kb에서 CNN보다 평균 MCC가 0.0060 높았지만 2/3 seed에서만 높았고, 차이가 작았다. LoRA는 FNO보다 5kb 평균 MCC가 0.0754 높았으며 2/3 seed에서 높았다. FNO−LoRA 차이는 1kb +0.0202에서 5kb −0.0754로 바뀌었다. 이 차이의 변화량은 −0.0956이다.

길어질수록 LoRA 성능 자체가 개선됐다고 해석할 수도 없다. LoRA의 1kb→5kb 평균 MCC 변화는 +0.0084로 작고, 평균 ROC-AUC는 0.6323→0.6201로 낮아졌다. 5kb ROC-AUC는 CNN 0.5680, FNO 0.5824, LoRA 0.6201이었다. [전체 6개 지표와 개별 run 수치](long_epi_results/TABLES.md), [원시 수치 CSV](long_epi_results/all_runs.csv)를 함께 확인할 수 있다.

Test 평가 전에 정한 paired stratified bootstrap 5,000회는 다음과 같은 참고 구간을 줬다.

| 평균 MCC contrast | 관측 차이 | Bootstrap percentile 95% 구간 |
|---|---:|---:|
| 5kb FNO − CNN | +0.0060 | [−0.0453, +0.0561] |
| 5kb FNO − LoRA | −0.0754 | [−0.1582, +0.0085] |
| FNO−LoRA 차이의 5kb−1kb 변화 | −0.0956 | [−0.1908, +0.0019] |

모두 0을 포함한다. 따라서 관측 평균의 방향을 설명할 수는 있지만, 차이가 확정됐거나 두 방법이 동등하다고 주장할 수 없다. 이 구간은 고정된 3개 학습 모델과 이번 balanced test cohort에 조건부이며, 재학습 변동·공유 enhancer/promoter에 따른 pair 간 상관·염색체 표본 불확실성·다중 비교 보정을 포함하지 않는다. Test 256쌍에는 exact/RC 기준 enhancer 254개, promoter 247개가 있다. [Bootstrap 원본](long_epi_results/bootstrap.json)

## 비용과 FNO를 선택할 이유

5kb에서 실제 측정한 비용은 다음과 같다.

| 방법 | 평균 학습 초/run ± SD | 평균 학습 초/epoch | 학습·표현 추출 peak MiB | 전체 모델 추론 pair/s |
|---|---:|---:|---:|---:|
| Frozen + CNN | 40.4 ± 9.5 | 5.5 | 967.7 | 37.3 |
| Frozen + FNO | 85.7 ± 63.7 | 8.9 | 967.7 | 37.1 |
| LoRA | 606.7 ± 261.1 | 66.1 | 3,570.1 | 39.9 |

FNO의 cached 학습 시간은 LoRA보다 약 7.1배 짧았다. 다만 이 수치는 epoch 수가 서로 다른 조기 종료 run의 평균이며, 공유 feature 추출을 제외한다. 5kb 전체 train/dev/test feature를 한 번 추출하는 데 42.5초가 추가됐고, float32 feature payload 저장량은 4.345GiB였다. CNN과 FNO는 같은 cache를 공유했다. 세 길이의 cache payload 합계는 약 6.98GiB다.

시간에는 training loop를 포함하고, dev 평가·checkpoint 저장·모델 구성 시간은 제외한다. 동일 프로세스에서 순차 실행해 CUDA/FFT cache 및 실행 순서의 영향을 받는다. 특히 5kb FNO seed 42의 첫 epoch는 약 88.3초로, 초기 FFT 준비 비용이 평균과 SD에 포함됐다. Peak는 PyTorch allocated 메모리 기준이며 장치 전체 사용량이나 추론 단계 peak를 뜻하지 않는다. Frozen 방법은 표현 추출과 학습 중 큰 값을 사용하므로 CNN/FNO 메모리 곡선이 겹친다.

추론 속도는 batch 2, test 앞 128쌍에 대해 encoder·분류기·GPU 전송을 포함하고 tokenization과 disk 읽기를 제외했다. FNO도 추론 때 DNABERT-2 전체를 통과한다. 이번 5kb 측정에서는 FNO의 추론 속도 이점이 없었다.

**저렴한 frozen adaptation이라는 용도는 남지만, 이번 긴 입력 실험에서 FNO만의 선택 근거는 약하다.** CNN은 FNO와 비슷한 MCC에 더 짧은 학습 시간을 보였다. 이전 300bp promoter 실험에서 FNO가 CNN을 이겼던 결과가 이번 EPI task에 그대로 이어지지는 않았다. 정확도 우선의 다음 비교에는 LoRA를 유지하고, frozen 방법의 비교 기준에는 CNN도 계속 포함하는 것이 타당하다.

주 성능 지표는 LoRA 병합 전 BF16 평가이고, 속도는 병합 후 평가다. 병합 전/후 class가 바뀐 예측은 1kb에서 총 0개, 2kb에서 3개, 5kb에서 10개였다(각 길이 3×256개 예측). 5kb의 병합 후 LoRA 평균 MCC는 0.2066으로, 주 평가의 0.2171과 완전히 같지는 않지만 위의 정성적 해석은 바뀌지 않는다. 병합 전/후 확률과 지표를 모두 보관했다. [비용 및 병합 수치](long_epi_results/costs.json)

## 범위, 출처와 검증

이 결과는 하나의 작은 balanced subset, 3개 seed, 고정 LR, 제한된 epoch에서 얻은 탐색 결과다. 검증 곡선에 변동이 있고 일부 run은 최대 epoch까지 실행됐으므로 최적 성능이나 수렴을 보장하지 않는다. 현재 결과만으로 FNO의 모든 긴 서열 응용을 부정하거나, 긴 context의 정보가 생물학적으로 무의미하다고 해석할 수 없다.

![검증 MCC 학습 곡선과 선택 checkpoint](C:/Users/public-user/Documents/ChatGPT/FNO/docs/long_epi_results/learning_curves.png)

원본 GM12878 14,000쌍을 합친 뒤 새 split을 만들었다. 모든 길이에서 enhancer·promoter·연결 입력의 split 간 exact/RC 겹침은 0이다. Near-homology, shifted genomic overlap, chromosome 분리는 통제하지 못했다. 공식 split과 다르므로 논문 수치나 기존 300bp promoter 실험과 절대 점수를 직접 비교하면 안 된다.

데이터 근거는 [DNABERT-2 논문의 GUE+ EPI 설명](https://arxiv.org/html/2306.15006v2)과 [공개 GUE_v2 데이터 미러](https://huggingface.co/datasets/genomic-benchmarks/GUE_v2)다. 미러 revision은 `f1290ccd49d4c7f8014cac47c93779aed06a294b`이며, 세 compressed file의 SHA256이 해당 revision의 LFS hash와 일치했다. 공식 Google Drive 아카이브는 TLS 연결 실패로 byte 단위 대조하지 못했다. [데이터 매핑·감사 기록](long_epi_results/data_source.json), [LFS 검증](long_epi_results/lfs_verification.json)

27개 run의 원본/데이터/checkpoint hash, 동일 source pair의 길이별 입력 재구성, split 간 component exact/RC 분리, 원본 encoder 가중치 유지, 예측 row 순서, 6개 test 지표 및 평균/SD를 재검증했다. 병합 전/후 LoRA의 저장된 예측에서도 지표를 다시 계산해 대조했다. 전체 25개 모듈 테스트가 통과했다. [검증 기록](long_epi_results/verification.json), [테스트 로그](long_epi_results/pytest.txt)

Study ID: `1d6eb7bfd080cd0b266c1eada0ac8b4a721c40f7eff6457746af87277a5bc9b7`. 학습 checkpoint는 `runs/long_epi_pilot/`에 있으며, 보고용 예측·수치·출처는 [export manifest](long_epi_results/export_manifest.json)에 해시와 함께 보관했다. 전체 재현 절차는 [프로토콜](LONG_EPI_PROTOCOL.md)에 있다.
