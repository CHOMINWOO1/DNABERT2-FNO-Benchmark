# EPI 입력 정보 진단: 사전 프로토콜

이 프로토콜은 이번 실험의 모델 학습 및 새 test 평가 전에 고정한다. 목적은 기존 GM12878 EPI 문제에서 한쪽 서열의 특성만으로 분류가 가능한지, 두 서열의 정보 결합과 joint encoding이 추가 이득을 주는지 확인하는 것이다. FNO 구조 탐색은 이번 범위에 포함하지 않는다.

## 데이터와 미사용 평가 표본

이전 5kb 실험의 train 1,024쌍과 dev 256쌍을 순서까지 유지한다. 원본의 enhancer 3,000bp와 promoter 2,000bp를 사용한다. 새 test는 label별 512쌍, 총 1,024쌍이다. Seed 20260913으로 추출하며, 이전 EPI train/dev/test 및 추가 진단 holdout 1,024쌍의 모든 구성 서열 그룹을 제외한다. 그룹은 enhancer 600/1200/3000bp, promoter 400/800/2000bp crop의 exact/RC 일치와 전이적 연결로 정의한다. 추출 실패 시 자동으로 exclusion을 완화하지 않는다.

이는 같은 GM12878 공개 corpus 내부의 새 표본이다. 염색체 분리, 근접 상동성, 역할을 가로지르는 유전체 중복은 통제하지 않는다. Test 내부의 행은 구성 서열을 공유할 수 있다. 이미 평가한 두 test는 이번 모델 선택에 사용하지 않는다. Dev의 반복 사용에 따른 개발 적응 가능성은 남는다.

## DNABERT-2 입력 조건 다섯 가지

Encoder는 기존 pinned DNABERT-2 117M을 고정하고 eval 모드, BF16, extraction batch 2로 실행한다. Attention은 기존 PyTorch eager 구현이다. Encoder 출력의 pooling은 float32이며 padding/특수 token을 제외한다. Feature cache에는 pooling한 벡터만 저장한다.

| 이름 | 입력 및 pooling | 분류기 입력 차원 |
|---|---|---:|
| enhancer | Enhancer만 인코딩하고 masked mean | 768 |
| promoter | Promoter만 인코딩하고 masked mean | 768 |
| separate_pair | 각각 독립 인코딩·mean 후 enhancer, promoter 순서로 벡터 결합 | 1536 |
| joint_pair | 두 DNA 서열을 연결하여 인코딩한 뒤 각 영역을 따로 mean하여 결합 | 1536 |
| joint_global | 두 DNA 서열을 연결하여 인코딩한 뒤 전체 masked mean | 768 |

Joint pair와 separate pair는 동일한 분류기 구조, 같은 seed의 동일 초기값, 동일 학습 샘플 순서를 사용한다. Joint pair의 영역 경계는 fast tokenizer의 bp offset으로 정한다. 경계에 걸친 BPE token은 각 영역에 겹치는 bp 비율에 따라 분할 가중한다. 이것은 bp 균일 격자로 재표집하는 실험이 아니다. Joint global은 기존 연결 입력·전체 mean 방식의 진단용 MLP 기준 모델이며, 기존 FNO 또는 CNN 모델을 재현하는 조건은 아니다.

분류기는 LayerNorm → Linear → GELU → Dropout(0.1) → Linear(2)다. 단일/쌍 입력에 맞춰 hidden width를 조절하고, 전체 학습 parameter를 237,571의 1% 이내로 맞춘다. 입력 차원과 hidden width가 다르므로 단일 대 쌍 비교의 capacity를 완전히 같게 만들 수는 없다. 독립 인코딩 후 nonlinear MLP도 두 벡터의 상호작용을 학습할 수 있다. 따라서 joint encoding 이득이 곧 생물학적 enhancer–promoter interaction의 입증은 아니다.

## 학습과 dev 선택

다섯 neural 조건 모두 LR 0.0001/0.0003/0.001 × seed 42/43의 동일 예산을 사용한다. 두 seed의 best-dev MCC 평균으로 LR을 선택하며 동점이면 낮은 LR을 택한다. 선택 LR에 seed 44를 추가하여 각 조건의 최종 결과를 3 seed로 계산한다. 총 학습 수는 5 × (3 × 2 + 1) = 35회다.

공통으로 batch 32, accumulation 1, AdamW, weight decay 0.0001, cosine schedule, 최대 30 epoch, patience 6, MCC min_delta 0.0001, gradient clip 1을 사용한다. 모든 neural 조건에 동일하게 적용한다. 이전 FNO 실험과 학습 예산 및 분류기가 다르므로 이전 점수와 직접 전후 비교하지 않는다. 기존 학습 엔진과 epoch별 재개 기능을 사용하되, head 학습에는 encoder가 없고 추출 단계의 encoder hash를 별도로 검증한다.

## 단순 조성 기준 모델 여섯 가지

GC 비율과 4-mer 빈도 각각에 대해 enhancer 단독, promoter 단독, 두 요소의 feature 결합을 비교한다. GC는 G/C 염기 수를 전체 길이로 나눈다. 4-mer는 ACGT로만 이루어진 유효 window 빈도로 정규화하며, 두 요소 사이 경계를 가로질러 세지 않는다. Pair는 요소별 feature를 이어 붙인다.

StandardScaler와 L2 logistic regression을 사용한다. Scaler는 train에만 fit하며 C=0.001/0.01/0.1/1/10 중 dev MCC로 선택한다. 동점이면 낮은 C를 선택한다. 총 30회 fit, 최종 6개 deterministic 모델이다. 수렴 warning은 오류로 처리한다. Neural과 동일한 탐색 공간이나 연산 예산을 갖는 비교는 아니며, 쉬운 조성 신호의 존재를 진단하는 기준이다.

## 평가와 해석

모든 학습 및 선택 완료 후 새 test 특징을 추출하고 최종 neural 15개와 조성 6개를 평가한다. 총 21회다. Threshold는 고정 0.5이고 test에서 보정하지 않는다. MCC를 주요 지표로, ROC-AUC, PR-AUC, average precision, F1, accuracy를 저장한다. Neural 평균 ± 표본 SD를 기록하며 deterministic 기준 모델에 인위적인 seed SD를 붙이지 않는다.

사전 관심 대비는 separate_pair − enhancer/promoter, joint_pair − separate_pair, joint_pair − enhancer/promoter, joint_pair − joint_global, kmer4_pair − kmer4_enhancer/promoter, joint_pair − kmer4_pair, joint_pair − gc_pair다. 각각 같은 test 행을 함께 재표집하는 label-stratified paired bootstrap 5,000회(seed 20260914)로 MCC 차이의 조건부 95% 구간을 계산한다. Neural은 세 seed별 MCC의 평균을 사용하며 ensemble MCC가 아니다. 재학습 변동과 test 행 간 구성 서열 공유 상관은 반영하지 않는다. 여러 대비를 탐색적으로 보고하며 다중 비교 보정은 하지 않는다. 0을 포함하는 구간을 동등성의 증거로 취급하지 않는다.

한쪽 단독이 결합과 비슷하면 이 데이터·표현·학습 예산에서 두 요소 결합의 추가 이득을 확인하지 못한 것으로 해석한다. GC/k-mer 성능이 높더라도 데이터 누출이나 artifact를 입증한 것은 아니다. Joint encoding 이득이 없더라도 실제 생물학적 상호작용의 부재를 뜻하지 않는다. 결과에 따라 다음 연구의 데이터와 표현 설계 우선순위를 결정한다.
