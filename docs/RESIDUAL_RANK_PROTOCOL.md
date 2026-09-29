# 고정 거리 점수 + 서열 보정: 분류 loss와 순위 loss 진단

이 프로토콜은 새 모델의 test 점수를 보기 전에 고정한다. 목적은 이전 frozen DNABERT-2 mixer 비교의 낮은 거리 통제 순위 성능에 학습 목표의 불일치가 기여했는지 진단하는 것이다. 원인으로 단정하지 않는다. 이번 실행은 앞서 제안한 작은 순위 학습 진단이며 연속 장거리 DNA 실험은 포함하지 않는다.

## 자료와 평가 분리

검증된 기존 K562 CRISPR 10,273행, 5개 chromosome fold, 10bp grid cache를 그대로 재사용한다. Outer test=f, inner dev=(f+1)%5, 나머지 세 fold가 train이다. 자료·fold·거리 비율 1.25의 397개 비교(45 promoter/45 gene)를 바꾸지 않는다. 이미 본 자료를 사용하는 탐색적 OOF 분석이며 새 독립 외부 검증이 아니다.

Fold마다 이전 실험에서 train 세 fold에만 fit한 distance-only StandardScaler + balanced logistic regression(C=0.001)을 hash 검증 후 재사용한다. 그 계수와 정규화는 고정한다. 기존 신경망 checkpoint는 재사용하지 않는다. 모델 입력에 test label, gene ID, assay effect size를 넣지 않는다.

## 보정 모델과 대조군

예측 logit = 고정 거리 logit + gamma × 서열 보정이다. 서열 보정은 enhancer/promoter의 각 1kb를 독립 인코딩한 frozen DNABERT-2 grid만 받으며 거리는 입력받지 않는다. 공유 residual mixer를 각 구간에 적용하고 mean pooling → E/P concat → LayerNorm → Linear(1536,64)/GELU/dropout/Linear(64,1)로 scalar 보정을 얻는다. 마지막 Linear의 weight와 bias를 0으로 초기화하여 초기 예측은 거리 기준 모델과 정확히 같다. Gamma는 학습 parameter가 아니다.

MLP/CNN/FNO는 이전 10bp grid mixer와 같은 width 79/62/44, CNN circular kernel 9, FNO modes 16, residual alpha 초기값 0.1을 사용한다. 동일 seed의 head 초기값을 공유한다. 학습 parameter는 약 23.7만 개이며 width 차이는 남는다. 전체 encoder는 학습하지 않는다. 마지막 bias는 pairwise loss에서 상쇄되는 비식별 parameter이며 parameter 수에 포함한다.

각 mixer마다 pointwise와 pairwise 두 objective를 새로 학습한다. **두 objective 모두 동일한 train-fold matched positive–negative 비교 쌍만 사용**한다. Train에는 27개 주요 promoter가 있다. 이는 이전 전체 행 분류 학습과 다르므로 이전 점수와 비교해 loss만의 효과라고 해석하지 않는다. 여기서의 통제 비교는 새로 학습한 두 objective 사이에서 수행한다.

- Pairwise: softplus(−(positive logit − negative logit)).
- Pointwise control: [softplus(−positive logit) + softplus(negative logit)] / 2.
- 각 promoter가 같은 총 가중치를 갖도록 pair weight = 전체 pair 수 / (promoter 수 × 해당 promoter의 pair 수)로 정한다. 비교 후보 재사용과 label 균형은 두 objective에서 같다. 추가 class weight는 없다.
- 양성/음성에 항상 같은 상수를 더하면 pairwise loss가 변하지 않는다. 순위 학습 logit을 보정된 확률로 해석하지 않는다.

## 고정 예산과 선택

3 mixer × 2 objective × 5 fold × 3 seed(42/43/44) = **90회 학습**이다. LR 0.0003을 공통으로 고정하고 새 LR 탐색은 하지 않는다. AdamW(weight decay 0.0001), cosine schedule, 최대 40 epoch, patience 8, min_delta 0.0001, pair batch 16 × accumulation 2, clip norm 1, dropout 0.1을 사용한다. BF16 matmul, FFT float32/complex64, deterministic PyTorch, TF32 off다. 같은 fold/seed의 pair 순서를 공유한다.

Checkpoint는 gamma=1인 dev의 주요 matched concordance로 선택한다. 최초 학습 epoch부터 후보로 삼고, min_delta보다 큰 향상이 없으면 더 이른 checkpoint를 유지한다. 학습 종료 뒤 그 checkpoint의 dev 예측에 gamma={0,0.25,0.5,1}을 적용한다. 최대 concordance를 선택하고 정확한 동점이면 작은 gamma를 택한다. 따라서 선택된 dev 점수는 거리 단독보다 낮지 않지만 **test 개선이나 성능 보존은 보장하지 않는다.**

모든 90회 학습·checkpoint·gamma 선택이 끝난 뒤에만 outer-test 점수를 계산한다. 90개 신경망 checkpoint와 5개 고정 거리 모델을 평가한다. 각 신경망의 같은 raw correction에서 gamma=1과 dev-selected gamma의 결과를 모두 저장한다. 이는 95개 fold별 모델 평가이며 추가 재학습은 아니다. 초기 출력 0, parameter 수, head 초기값, baseline hash, loss/gradient, fold 분리와 예측 재현을 검증한다.

## 보고할 지표와 판단

주요 지표는 비율 1.25에서 promoter별 평균 후 45개 promoter를 동일 가중 평균한 concordance다. 동점은 0.5다. Float64 거리 logit에 보정을 더한 score로 순위를 계산하며 sigmoid 포화로 순위 동점이 생기지 않도록 확률 대신 score를 주요 순위 지표에 사용한다. 세 seed 평균 ± 표본 SD를 보고한다. Dev-selected gamma가 주요 배포 규칙이며, gamma=1도 함께 보고해 fallback으로 차이가 가려지는지 확인한다.

사전 지정 비교는 pairwise FNO−pairwise CNN/MLP/거리, 각 mixer의 pairwise−pointwise, pairwise MLP/CNN−거리다. 선택 gamma와 gamma=1 모두 같은 대비를 보고한다. Gene 45개를 paired bootstrap 10,000회(seed 20260919) 복원 추출하여 95% percentile 구간을 구한다. 먼저 seed별 promoter 점수의 평균을 낸다. 고정 학습 모델에 조건부이고 공유 train/fold 상관, 재학습 및 chromosome 불확실성 전체를 포함하지 않으며 다중 비교 보정은 없다.

고정 OOF 예측에 비율 1.1(32 loci)/1.5(52 loci) 민감도 분석을 적용한다. 집합이 달라짐을 명시한다. 모든 123 mixed-label promoter의 거리 제한 없는 macro AUROC도 보조로 기록한다. 모델 선택은 이 보조 점수에 맞춰 바꾸지 않는다.

Pairwise가 pointwise보다 개선되면 이번 동일 pair 학습 조건에서 loss 선택의 영향으로 해석할 수 있다. 하지만 과거 전체 행 학습 실패의 유일한 원인을 증명하지 않는다. FNO가 CNN/MLP보다 우세하지 않으면 FNO 고유 이득을 주장하지 않는다. 거리 대비 추가 이득이 없으면 현재 자료·표현의 추가 튜닝을 종료한다. 탐색적 개선이 있어도 확인 주장은 새 외부 평가가 필요하다.
