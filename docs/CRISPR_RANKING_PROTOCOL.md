# CRISPR 후보 enhancer 순위 평가: 사전 프로토콜

모델 학습 전에 고정한다. 목적은 같은 promoter에 대해 실험적으로 지지되는 enhancer 후보를 구분할 수 있는지, enhancer 서열과 실제 거리 정보가 무엇을 추가하는지 확인하는 것이다. FNO를 추가하거나 긴 연속 서열을 학습하는 실험은 이번 범위에 포함하지 않는다.

## 공개 데이터와 좌표

EngreitzLab/CRISPR_comparison의 commit `50587422e6b11259ead6fbc6f867681c788f39b7`에 있는 `EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz`를 사용한다. 연구실이 정리한 K562 CRISPR 표로, 원본 10,356쌍 중 Regulated=TRUE는 471쌍이다. Nasser2021, Gasperini2019, Schraivogel2020을 포함한다. 다운로드 파일을 해당 Git blob SHA-1 및 로컬 SHA-256으로 검증했다. 이 파일의 `training_K562`는 원 연구의 명칭이며, 이번에는 내부를 새 염색체 분할로 나눈다. 원 논문의 공식 점수를 재현하는 실험이 아니다.

Label은 원본 Regulated를 사용한다. Significant=TRUE이면서 EffectSize<0인 것과 일치함을 확인한다. Negative label은 이 curated assay의 음성 판정이며 생물학적 효과가 절대 없다는 의미가 아니다. EffectSize, p-value, power, direct-effect probability, dataset ID, 유전자 ID는 모델 feature로 넣지 않는다. Gene ID는 그룹 평가 및 분할 검증에만 사용한다.

Enhancer는 제공된 perturbed interval의 정수 midpoint를 중심으로 1,000bp, promoter는 startTSS를 중심으로 1,000bp를 추출한다. TSS가 0으로 누락된 행, 두 입력 window가 겹치는 행, 한 window에서 N이 10%를 넘는 행을 제외하고 원본 행 번호와 이유를 기록한다. 이는 서로 다른 길이의 전체 perturbed interval을 모두 입력하는 방식은 아니다. 이웃 DNA를 포함하거나 긴 interval 일부만 포함할 수 있다.

거리 feature는 GRCh38 enhancer interval midpoint와 startTSS 사이의 절댓값을 다시 계산한다. 원본 distanceToTSS는 좌표에서 재계산한 값과 차이가 있어 feature로 사용하지 않는다. Ensembl GRCh38 sequence API에서 reference-positive 방향 서열을 얻고 반환된 assembly·좌표·길이를 검증한다. 6개 분산된 window를 UCSC hg38 API와 교차 확인한다. 원본에 strand가 없어 양방향 유전체 좌표에서 대칭 promoter window를 쓰며, K562 개별 haplotype은 반영하지 않는다.

## 염색체 분할과 평가 단위

- Test: chr1, chr3, chr5.
- Dev: chr2, chr4, chr6.
- Train: 나머지 염색체.

분할은 성능 확인 전에 고정한다. 동일 유전자·promoter·enhancer 좌표가 split을 넘지 않는지 검사한다. 서로 다른 염색체에 같은 1kb 서열 또는 RC alias가 있으면 train → dev → test 순서로 뒤쪽 split의 해당 행을 제외한다. Enhancer와 promoter 역할을 가로지르는 exact/RC alias도 검사한다. 근접 상동성은 통제하지 않는다.

주요 평가는 `(chrTSS, startTSS, gene ID)`가 동일한 promoter locus 안에서 양성·음성 후보가 모두 있는 그룹의 AUROC를 구한 뒤, locus별 동일 가중치로 평균한 **within-promoter macro AUROC**다. 후보가 한 label로만 구성된 그룹은 전체 분류 평가에는 포함하지만 이 순위 지표에서는 제외한다. 몇몇 gene은 TSS가 여러 개이므로 gene ID만으로 서로 다른 promoter 입력을 섞지 않는다.

보조 지표는 같은 방식의 macro average precision(AP), macro AP minus group positive fraction, tie-aware top-1 hit rate, 전체 행의 AP/ROC-AUC/MCC/F1/accuracy/PR-AUC다. Top-1 동점이면 최고점 후보들의 양성 비율을 그 그룹의 점수로 사용한다. 모든 후보를 같은 점수로 주는 promoter-only 모델의 within-promoter AUROC는 정의상 0.5다. 이 값 자체가 새로운 생물학적 발견은 아니다.

## 모델과 공정한 탐색

Frozen DNABERT-2의 기존 pinned revision으로 각 1kb window를 **독립적으로** 인코딩하고, DNA token만 float32 masked mean한다. BF16 encoder, extraction batch 4를 사용한다. 유일한 서열당 특징을 한 번만 추출해 같은 promoter의 입력 벡터를 일치시킨다.

Neural 조건은 promoter-only, enhancer-only, enhancer+promoter, enhancer+distance, enhancer+promoter+distance의 5개다. 두 DNA 벡터는 enhancer, promoter 순서로 결합한다. 거리의 log10을 train 평균·표준편차로 정규화한다. DNA feature에 LayerNorm을 적용한 뒤 거리 scalar를 붙이고 Linear → GELU → Dropout(0.1) → Linear(2)로 분류한다. 각 조건의 학습 parameter를 237,571의 1% 이내로 맞춘다. 입력 차원과 hidden width의 차이는 남으므로 capacity가 완전히 동일하다고 주장하지 않는다.

각 조건마다 LR 0.0001/0.0003/0.001 × seed 42/43을 탐색한다. 두 seed의 best-dev macro AUROC 평균으로 LR을 고르고 동점이면 낮은 LR을 선택한다. 선택 LR에 seed 44를 추가한다. Neural 학습은 총 35회이며, 최종 15개 모델을 평가한다. Promoter-only는 선택 지표가 항상 0.5이므로 같은 동점 규칙이 적용된다.

공통 AdamW, weight decay 0.0001, cosine schedule, batch 64, accumulation 1, 최대 30 epoch, patience 6, min_delta 0.0001, gradient clip 1이다. 불균형 대응은 train label 수로 고정한 class weight `N/(2*N_class)`의 cross entropy를 사용한다. 손실 합을 실제 sample 수로 나눈다. Test에서 threshold를 보정하지 않고 0.5로 고정하므로 전체 분류 지표는 class weighting에 영향을 받는다. 주요 지표는 threshold와 무관한 후보 순위다.

기준 모델은 log10(distance) 단독, 두 요소 GC+distance, 두 요소 4-mer+distance의 StandardScaler + balanced L2 logistic regression이다. Scaler는 train에만 fit하며 C=0.001/0.01/0.1/1/10 중 dev macro AUROC로 선택한다. 총 15회 fit, 최종 3개 모델이다. 4-mer는 요소별로 세며 두 요소 경계를 가로지르지 않는다. 이 기준 모델은 nonlinear neural model과 표현력이 같지 않다.

Inference에서는 동일 input feature vector를 한 번만 평가하고 확률을 원래 행에 되돌린다. GPU batch shape의 미세한 반올림 차이가 같은 promoter 안에서 가짜 순위를 만들지 않도록 하기 위함이다. Test label은 이 처리에 사용하지 않는다.

## 평가 시점, 불확실성과 범위

모든 학습과 dev 선택이 끝난 뒤 test DNA 특징을 추출하고 neural 15개, logistic 3개를 평가한다. 전체 18회다. 모델과 임계값을 test 결과에 맞춰 변경하지 않는다.

사전 관심 대비는 pair−promoter, pair−enhancer, enhancer_distance−enhancer, pair_distance−pair, pair_distance−enhancer_distance, pair_distance−distance, enhancer_distance−distance, pair_distance−kmer4_distance다. Test의 mixed-label promoter loci에서 seed별 AUROC를 평균한 뒤 차이를 계산한다. Gene을 복원 추출하여 해당 gene의 모든 eligible promoter loci를 함께 재표집하는 cluster bootstrap 5,000회(seed 20260916)의 percentile 95% 구간을 보고한다. 각 재표집에서는 promoter locus 평균을 유지한다. 재학습 변동, 3개 test 염색체 밖의 변동, 다중 비교 보정은 반영하지 않는다.

Promoter 단독을 넘는 성능은 같은 promoter의 enhancer 후보를 구분한다는 뜻이다. 그러나 거리 기준 모델도 후보를 구분할 수 있으므로 **거리 대비 추가 이득**을 따로 확인해야 한다. Pair가 enhancer-only를 넘지 않으면 promoter와의 조합에 특화된 이득을 입증하지 못한 것이다. K562 한 cell type의 reference sequence 기반 실험이며, chromatin state, 실제 3D contact 또는 intervening sequence를 관찰하지 않는다. 단독으로 causal mechanism이나 FNO의 유용성을 입증하지 않는다.

## 출처

- [연구실 benchmark와 형식 설명](https://github.com/EngreitzLab/CRISPR_comparison/tree/50587422e6b11259ead6fbc6f867681c788f39b7/resources/crispr_data)
- [CRISPR 데이터 통합 코드](https://github.com/argschwind/ENCODE_CRISPR_data)
- [UCSC 서열 API의 좌표 규칙](https://genome.ucsc.edu/goldenpath/help/api.html)
