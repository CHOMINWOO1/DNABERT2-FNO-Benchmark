# 거리 통제 MLP/CNN/FNO 비교: 사전 프로토콜

모델 성능을 확인하기 전에 고정한다. 목표는 같은 promoter 안에서 거리가 비슷한 양성·음성 후보를 구분하는 데 FNO가 MLP, CNN 또는 거리 단독보다 추가 이득을 주는지 확인하는 것이다.

## 표본과 탐색적 교차 평가

기존 좌표 검증을 통과한 K562 CRISPR 10,273행을 재사용한다. 주요 비교는 같은 promoter의 양성·음성 두 후보 간 거리의 최대/최소 비율이 **1.25 이하**인 모든 비교다. 성능을 사용하지 않은 feasibility audit에서 45개 promoter·45개 gene, 397개 비교, 양성 행 81개와 음성 행 196개를 확보했다. 후보 행이 여러 비교에 반복되므로 397개 독립 표본으로 취급하지 않는다. 비율 1.1과 1.5는 사전 고정한 민감도 분석이며 각각 32개와 52개 promoter다. 2.0은 feasibility 수만 기록하고 주결론 선택에 사용하지 않는다.

기존 23개 promoter test에 반복 적응하는 것을 피하기 위해, 모든 chromosome을 5개의 서로 겹치지 않는 fold로 나눠 out-of-fold 예측을 만든다. 각 fold에 주요 거리 통제 promoter가 9개가 되도록 label·거리 기반 표본 수만으로 묶었다. 데이터와 모델 설정은 이전 연구 과정에서 이미 관찰했으므로 **새 외부 확인 실험이 아닌 탐색적 교차 평가**다.

| Fold | Chromosomes | 전체 행 | 양성 | 주요 비교 promoter | 주요 비교 수 |
|---|---|---:|---:|---:|---:|
| 0 | chr19/13/14/15/16 | 2561 | 69 | 9 | 48 |
| 1 | chr11/20/9 | 1610 | 85 | 9 | 44 |
| 2 | chr7/8/12/10 | 1797 | 89 | 9 | 230 |
| 3 | chrX/1/3/22 | 2690 | 104 | 9 | 60 |
| 4 | chr5/2/17/18/4/6/21 | 1615 | 123 | 9 | 15 |

Outer test=f, inner dev=(f+1)%5, 나머지 세 fold가 train이다. 각 행의 최종 예측은 그 chromosome이 학습과 dev에 없던 모델에서만 얻는다. Gene ID, 같은 입력 서열 및 RC alias가 fold를 넘는지 검증한다. 모든 fold의 모델 선택을 마친 뒤 outer-test 지표를 계산한다.

## 동일한 입력과 mixer 비교

Frozen DNABERT-2의 기존 고정 revision으로 enhancer와 promoter의 1kb reference window를 각각 독립 인코딩한다. BPE token의 bp offset을 이용해 hidden state를 **10bp 단위 100개 bin**으로 평균한다. Token이 bin 경계를 가로지르면 겹치는 bp 길이 비율로 기여한다. 특수 token·padding은 제외한다. 이 과정은 float32이며 cache를 float32로 저장한다. Encoder는 BF16, 추출 batch 4다.

세 모델 모두 같은 grid cache를 사용한다. 길이·label에 따라 학습되는 변환이 없으므로 고정 encoder의 모든 서열 특징을 사전 계산할 수 있다. Test label을 학습·정규화에 사용하지 않는다. 학습 시 encoder는 없고 GPU의 고정 feature tensor를 조회한다. 이 방식은 기존 BPE token mean과 다르므로 이전 실험 점수와 직접 전후 비교하지 않는다.

각 1kb 구간에 같은 가중치를 공유하는 residual adapter를 적용한다: LayerNorm → down projection → mixer + pointwise branch → GELU/LayerNorm/dropout → up projection → residual(alpha=0.1 초기화). 그 뒤 enhancer와 promoter를 각각 mean pooling해 두 벡터를 결합하고, train에서 표준화한 log10(distance)를 붙인다. 공통 64-hidden nonlinear classifier로 예측한다.

- MLP: token별 1×1 channel mixing. 추가적인 위치 간 mixing이 없다.
- CNN: circular kernel 9, 즉 grid에서 인접 90bp 범위를 섞는 branch.
- FNO: Fourier modes 16의 spectral branch. FFT는 float32/complex64로 계산한다.

MLP는 이전의 pooled-vector MLP와 다른 **token별 residual control**이다. 세 모델의 전체 학습 parameter가 237,571의 1% 이내가 되도록 adapter width를 고르고, 공통 classifier는 같은 seed에서 동일한 초기값을 사용한다. Width 차이가 남으므로 mixer 연산 하나만 바꾸는 완전한 단일 요인 비교는 아니다. 구조 후보를 별도로 탐색하지 않는다.

FNO와 CNN은 각 1kb 구간 안에서만 작동한다. 두 enhancer–promoter 사이의 실제 연속 DNA나 genomic gap을 Fourier 축으로 연결하지 않는다. 이번 실험은 frozen representation의 구간별 augmentation이며, 장거리 유전체 backbone을 검증하는 실험이 아니다.

## 학습·선택 예산

모델 3개 × outer fold 5개 각각에 LR 0.0001/0.0003/0.001 × seed 42/43을 동일하게 탐색한다. Dev의 거리 통제 matched concordance를 checkpoint와 LR 선택에 사용한다. 두 seed의 best-dev 평균이 높은 LR을 고르고, 동점이면 낮은 LR을 택한다. 선택 LR에 seed 44를 추가한다. 총 **105회 학습**, 최종 neural checkpoint **45개**다.

공통 AdamW, weight decay 0.0001, cosine schedule, dropout 0.1, 최대 20 epoch, patience 4, min_delta 0.0001, gradient clip 1이다. Batch 32 × accumulation 2로 effective batch 64를 사용한다. Train fold의 label 수로 class weight N/(2*N_class)를 계산한 weighted cross entropy를 사용한다. 거리 평균·표준편차도 해당 train fold에만 fit한다. Dev가 9개 주요 promoter로 작으므로 튜닝 불확실성이 남는다.

거리 기준 모델은 fold마다 train의 log10(distance)에 StandardScaler + balanced logistic regression(C=0.001)을 fit한다. C는 이번 score를 보기 전에 고정한 기준값이다. 총 5개 distance fit이다. 모든 모델을 같은 outer-test 행에서 평가하므로 최종 fold별 평가는 **50회**다.

## 지표와 불확실성

각 matched positive–negative 비교에서 positive 점수가 높으면 1, 낮으면 0, 동점이면 0.5를 준다. 먼저 각 promoter 안에서 평균한 뒤 45개 promoter를 같은 가중치로 평균한다. 이를 **거리 통제 concordance**로 부른다. 거리 제약이 없는 AUROC와 동일한 지표로 부르지 않는다. Caliper가 1.25라도 잔여 거리 차이는 남으므로 같은 비교에서 distance-only 모델을 평가한다.

주요 모델 값은 세 seed의 OOF concordance 평균 ± 표본 SD다. Gene 45개를 복원 추출하는 paired cluster bootstrap 10,000회(seed 20260918)의 95% percentile 구간으로 FNO−CNN, FNO−MLP, FNO−distance를 보고한다. 보조로 CNN−MLP와 MLP−distance도 기록한다. 공유된 학습 자료로 인한 fold 간 상관, 재학습 및 chromosome 선택의 불확실성은 포괄하지 않으며 다중 비교 보정은 하지 않는다.

민감도 분석은 고정된 동일 OOF 확률에 거리 비율 1.1/1.5를 적용한다. 모든 mixed-label promoter의 일반 within-promoter AUROC도 보조 지표로 기록한다. 모델 또는 threshold를 이 분석 결과에 맞춰 변경하지 않는다.

FNO가 주요 조건에서 CNN/MLP/거리 대비 안정적인 이득을 보이지 않으면 현재 구간별 frozen DNABERT-2 augmentation 방향의 추가 튜닝 우선순위를 낮춘다. 이 결과를 다른 구조, 더 긴 연속 유전체 서열 또는 FNO 일반의 효과에 대한 결론으로 확장하지 않는다.
