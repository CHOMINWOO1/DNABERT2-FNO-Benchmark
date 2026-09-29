# 연속 64kb DNA의 장거리 정보 활용: 사전 프로토콜

새 유전자 발현 과제에서 CNN/FNO/attention의 구간 사이 정보 활용을 비교한다. 이 문서는 모델 학습·test 성능 계산 전에 고정한다. Frozen DNABERT-2가 독립 인코딩한 지역 표현을 전역 mixer로 연결하는 탐색적 실험이며 새로운 DNA foundation model의 pretraining이나 기존 attention layer 교체는 아니다.

## 자료와 좌표

공개 [Genomics LRB](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark/tree/b56f68ee30c01daf8f80a8d0b8a47f84199de08e/bulk_rna_expression)의 bulk RNA expression 자료를 사용한다. 고정 revision은 `b56f68ee30c01daf8f80a8d0b8a47f84199de08e`이며 CC BY-NC-SA 4.0 출처를 기록한다. 원 자료는 ExPecto에서 가공한 GTEx/FANTOM5 기반 218개 조직·세포 유형의 발현값이다. 개별 참가자 정보나 개인 유전체를 사용하지 않는다. 좌표 CSV와 label CSV의 행 대응은 공개 loader와 동일하게 유지하며 해당 loader는 실행하지 않는다.

원래 chr8 test를 유지하고 원래 train 중 chr7/chr9를 dev로 분리한다. 나머지 chromosome이 train이다. 64kb 경계 검사를 먼저 하고 gene ID·좌표의 고정 SHA 순서(seed 20260920)로 train 최대 2,048, dev 최대 512를 선택한다. Test는 모든 경계 적합 행을 사용한다. 발현 label 또는 모델 성능으로 행을 선택하지 않는다. 64kb 서열의 N 비율이 1%를 넘는 행은 제외하고 보충 추출하지 않는다.

최종 **train 2,030 / dev 502 / test 982개 gene**, 출력 218개다. 공개 파일의 Git blob/LFS SHA-256을 검증했다. GRCh37 Ensembl API에서 각 CAGE 대표 TSS 주변의 **실제 연속 65,536bp**를 얻었다. 원래 TSS는 1-based이며 저장 좌표는 0-based half-open이다. 음성 strand는 전체 서열을 reverse-complement하여 공개 loader와 같은 방향을 사용한다. 세 split × 두 strand의 6개 창을 UCSC hg19와 전장 exact-match 검증했다. 이것은 모든 창의 독립 교차 검증은 아니다. 모든 API 응답과 hash를 저장한다.

64kb 창이 겹치는 test 유전자들은 connected interval component **533개**로 묶는다(최대 20 gene). Bootstrap에서는 이 묶음을 재표집한다. Train/dev/test chromosome과 gene은 겹치지 않는다. Whole-window exact/RC 분리를 확인하고 최종 검증에서 1kb chunk 및 중앙 view의 exact/RC alias도 점검한다. 동일하거나 RC인 저복잡도 chunk가 다른 chromosome에 나타날 수 있으므로, 있다면 개수와 범위를 보고하고 임의로 데이터를 재선택하지 않는다. Near-homology는 검증 범위 밖이다.

자료에 제공된 label은 이미 log1p 후 표준화돼 있다. 모델 학습 전에 선택된 train gene에서만 조직별 평균·표준편차를 다시 fit한다. 원 자료에 적용된 조직별 affine 정규화는 이 재표준화로 상쇄되지만, 공개 label 처리 외의 숨은 전처리를 독립 재현한 것은 아니다. Test label은 학습·정규화·모델 선택에 사용하지 않는다.

## 지역 인코딩과 전역 구조

DNABERT-2 revision `7bce263b15377fc15361f52cfab88f8b586abda0`을 freeze한다. 연속 64kb를 **64개 1,024bp chunk**로 나누어 독립 인코딩한다. BPE offset과 겹치는 bp 길이를 가중치로 하여 chunk별 4개 256bp bin을 float32 평균한다. 특수 token/padding은 제외하고 입력을 자르지 않는다. Encoder BF16, feature cache float32다. 지역 encoder는 chunk 경계를 넘어 attention하지 않는다.

따라서 전체 입력은 [256 bin, 768 feature]이며, 같은 cache의 중심을 잘라 **4,096 / 16,384 / 65,536bp**를 비교한다. 세 길이에서 같은 gene·label을 사용한다. 전역 mixer에 들어가는 bin 위치는 TSS로부터의 상대 위치를 나타내는 고정 sinusoidal encoding으로 보존한다.

공통 입력 LayerNorm → down projection → 3개 residual mixer/FFN block → LayerNorm → 중심 **1kb(4 bin) readout** → 공통 64-hidden/218-output head를 사용한다. 평균 pooling은 전체 서열이 아니라 중심 1kb에서만 수행한다. 원거리 정보가 결과에 기여하려면 전역 block을 통해 중심으로 전달돼야 한다.

- CNN: kernel 17, dilation 1/4/16, zero padding. 3개 block의 이론적 receptive field 337 bin(86,272bp)으로 64kb 전체를 덮는다.
- FNO: learned Fourier modes 8 + pointwise branch. Float32/complex64 FFT, 주기적 Fourier 경계 가정이 남는다.
- Attention: 4-head self-attention, 같은 3개 block/FFN 구조.

학습 parameter 목표는 30만 개의 ±1%다. CNN/FNO width 61, 각각 **302,592개**이고 attention width 92, **298,762개**다. Head는 동일 seed에서 같은 초기값을 사용한다. Attention width와 mixer의 inductive bias 차이가 있어 완전한 단일 연산 교체 비교는 아니다. CNN이 원거리 위치를 보지 못하도록 설계하지 않았다. 비교에서 층 수·kernel·mode 수를 추가 탐색하지 않는다.

## 학습과 모델 선택

3 model × 3 length마다 seed 42에서 LR 0.0001/0.0003을 같은 예산으로 탐색한다. Dev의 조직별 gene 간 Pearson correlation을 218개 조직에 동일 가중 평균한 **macro Pearson**으로 best checkpoint와 LR을 선택한다. Epoch는 min_delta 0.0001, LR 동점은 작은 값을 우선한다. 선택 LR로 seed 43/44를 추가한다. 총 **36 training trial**, 최종 **27 checkpoint**다.

AdamW(weight decay 0.0001), MSE over train-standardized 218 outputs, cosine schedule, 최대 40 epoch, patience 6, batch 16 × accumulation 2, gradient clip 1, dropout 0.1을 공통으로 사용한다. 같은 seed의 sample 순서를 모델과 길이 사이에서 맞춘다. Encoder는 학습하지 않는다. Torch deterministic, BF16 matmul, FFT float32/complex64, TF32 off다. 모든 36회 학습과 LR 선택이 끝난 뒤 test를 평가한다.

주요 지표는 test의 **조직별 gene 간 Pearson을 평균한 값**이다. Secondary는 train-standardized target의 MSE 및 조직별 Pearson이다. 분산이 0인 예측 또는 target의 Pearson은 계산상 0을 부여하고 해당 개수를 함께 보고한다. 이 정의를 보고 없이 NaN 제거로 바꾸지 않는다. Train mean predictor는 MSE 기준선으로 기록하며 발현 확률·임상 예측으로 해석하지 않는다.

## 원거리 정보 진단

64kb에서 학습한 최종 9개 checkpoint에 대해 중심 **4kb를 그대로 유지**하고 다음 feature-space perturbation을 적용한다. 재학습하거나 test에 맞춰 강도를 선택하지 않는다.

1. **Mask:** 바깥 60kb의 각 feature bin을 train feature의 전역 평균 벡터로 대체한다. Model parameter·위치 좌표·입력 길이는 유지한다. DNA deletion을 모사하는 생물학적 개입은 아니다.
2. **Shuffle:** 바깥 60개의 1kb chunk를 재배열한다. 각 chunk 내부 4개 bin의 순서를 유지하므로 지역 DNABERT context는 손상시키지 않는다. 중심 4개 chunk는 제자리에 둔다. Label과 무관한 5개 고정 seed에 대해 모든 test gene에 같은 permutation을 적용한다.
3. **Swap:** 각 gene의 바깥 feature를 다른 test gene의 같은 위치 바깥 feature로 대체한다. 중심 4kb는 원래 gene 것을 유지한다. 고정 random cycle로 자기 자신을 donor로 쓰지 않으며, 5개 seed를 사용한다. Donor 선택에 label을 사용하지 않는다.

5개 seed는 20260921–20260925다. Shuffle/swap의 주요 진단 값은 각 학습 seed에서 5개 perturbation의 **성능 평균**이며 예측 ensemble 성능과 구별한다. 총 평가 조건은 intact 27개, mask 9개, shuffle 45개, swap 45개로 **126개 prediction matrix**다. 64kb 대비 4kb 재학습 비교도 있어 mask의 분포 변화만으로 결론을 내리지 않는다. 추가 진단은 입력 의존성을 보여줄 뿐 enhancer의 인과적 역할이나 특정 생물학적 기전을 입증하지 않는다.

## 불확실성과 판단 기준

주요 모델 점수는 3개 training seed 평균 ± 표본 SD다. Paired interval-cluster bootstrap 2,000회(seed 20260926)로 test의 533개 overlapping-window component를 복원 추출한다. 각 추출에서 포함된 gene을 모아 조직별 Pearson과 macro를 재계산하고, 같은 표본을 모든 비교에 사용한다. Training seed의 성능을 평균한다. Shuffle/swap은 perturbation seed의 성능도 평균한다. 중복 뽑힌 component의 gene은 반복 포함한다. 큰 cluster의 gene 수는 유지되어 지표 자체는 gene 단위다.

사전 지정 대비는 64kb FNO−CNN/attention, 각 모델의 64kb−4kb 및 64kb−16kb, 각 64kb 모델의 intact−mask/shuffle/swap이다. 총 17개 대비를 모두 보고하며, 유리한 조직이나 perturbation을 사후 선택하지 않는다. Bootstrap은 고정 모델과 chr8에 조건부다. 재학습, 다른 chromosome과 다른 조직 선택의 불확실성을 포괄하지 않고 다중 비교 보정도 없다.

긴 입력에서 개선되고 원거리 perturbation에서 일관되게 성능이 감소하면 **이번 모델의 예측이 원거리 입력 정보에 의존한다는 근거**로 해석한다. FNO의 고유 이득은 CNN 및 attention과의 같은 조건 비교에서 따로 확인해야 한다. 이 두 조건이 충족되지 않으면 FNO의 장거리 장점을 주장하지 않는다. 성능이 비슷하더라도 실제 end-to-end encoder 비용을 포함한 길이별 비용은 별도로 필요하다.

시간·GPU 메모리는 encoder 추출, cache 사용 학습, cache 기반 추론을 구분해 저장한다. 이 실험의 attention은 최대 256개의 지역 표현에 작동하므로 DNABERT 전체를 64kb로 확장한 attention 비용과 비교한 것이 아니다. 고정 feature cache 사용 속도를 전체 DNA 추론 속도로 표시하지 않는다. 기존 CRISPR 점수와 새로운 발현 Pearson을 직접 비교하지 않는다. 제한된 train subset·고정 구조·하나의 test chromosome에서 얻는 탐색 결과다.
