# CRISPR enhancer 후보 순위 실험 결과

2026-09-09. **같은 promoter 안에서 enhancer 후보를 구분하는 평가를 구성했지만, DNABERT-2 서열 정보가 거리 기준 모델을 안정적으로 넘는다는 근거는 얻지 못했다.** 서열 두 개와 거리를 사용한 모델의 평균 within-promoter AUROC는 0.7285, 거리만 사용한 모델은 0.7115였다. 차이 +0.0170의 유전자 단위 조건부 95% bootstrap 구간은 [−0.0582, +0.0964]였다.

공개 CRISPR 실험 표와 GRCh38 좌표를 사용하고, 학습에 포함하지 않은 chr1/3/5에서 평가했다. GPU에서 신경망 35회 학습, logistic regression 15회 fit, 최종 18회 평가를 완료했다. 코드 테스트 35개와 모든 최종 예측·지표 재검증이 통과했다.

## 무엇을 평가했나

이전 GM12878 분류 실험과 달리 이번 데이터는 K562 CRISPR assay의 enhancer–gene 후보 쌍이다. Enhancer를 억제했을 때 유전자 발현이 유의하게 감소한 것으로 curated된 후보를 양성으로 사용한다. 같은 promoter에 대한 양성·음성 후보의 점수 순서를 비교하므로, promoter 자체의 특성만으로 유전자 사이를 구분하는 효과를 주요 지표에서 통제한다.

Test 전체는 1,881쌍(양성 79쌍)이다. 그중 같은 promoter에 양성·음성 후보가 함께 있는 **23개 promoter·23개 유전자, 368쌍(양성 27쌍)**에서 주요 순위 지표를 계산했다. 각 promoter의 AUROC를 구한 뒤 동일 가중치로 평균한다. 나머지 한 종류 label만 있는 promoter는 전체 분류 지표에는 포함한다.

Promoter-only 모델은 같은 promoter의 모든 후보에 같은 점수를 주므로 **within-promoter AUROC가 정의상 0.5**다. 이전 실험에서 promoter 단독의 전체 분류 MCC가 높았던 사실과 모순되지 않는다. 두 지표가 다른 질문에 답하기 때문이다.

## 결과

신경망 값은 seed 42/43/44의 평균 ± 표본 SD다. Logistic 기준 모델은 deterministic fit 한 개의 결과다. AP는 average precision이며, within-promoter AP는 eligible promoter마다 계산해 동일 가중치로 평균했다. Global AP는 전체 test 1,881행에서 계산한다.

| 모델 | Within-promoter AUROC | Within-promoter AP | Global AP |
|---|---:|---:|---:|
| Promoter 단독 | 0.5000 ± 0.0000 | 0.3249 ± 0.0000 | 0.0957 ± 0.0032 |
| Enhancer 단독 | 0.5901 ± 0.0771 | 0.6215 ± 0.0546 | 0.0563 ± 0.0018 |
| Enhancer + promoter | 0.6288 ± 0.0411 | 0.6304 ± 0.0303 | 0.0868 ± 0.0066 |
| Enhancer + 거리 | 0.7019 ± 0.0090 | 0.6944 ± 0.0010 | 0.5045 ± 0.0212 |
| Enhancer + promoter + 거리 | **0.7285 ± 0.0364** | **0.7153 ± 0.0160** | 0.4303 ± 0.0431 |
| 거리 단독 | 0.7115 | 0.7100 | **0.5505** |
| GC + 거리 | 0.7077 | 0.7073 | 0.5504 |
| 4-mer + 거리 | 0.6982 | 0.7054 | 0.4764 |

![같은 promoter 내 CRISPR 후보 순위 비교](C:/Users/public-user/Documents/ChatGPT/FNO/docs/crispr_ranking_results/ranking_comparison.png)

서열 두 개에 거리를 추가하면 평균 AUROC가 0.6288에서 0.7285로 올라갔다. 그러나 거리 단독도 0.7115였고, 서열을 더한 이득은 평균 +0.0170에 그쳤다. Enhancer+거리 모델은 0.7019로 거리 단독보다 수치상 낮았다. **좋아 보이는 최종 점수를 서열 관계를 학습한 성과로 바로 해석할 수 없다.**

지표에 따른 차이도 있다. 같은 promoter 내 AUROC 평균은 두 서열+거리 모델이 가장 높았지만, 전체 후보에 대한 Global AP는 거리 단독이 더 높았다. 따라서 전반적으로 더 우수한 모델을 얻었다는 주장은 하지 않는다.

## 유전자 단위 불확실성

이미 학습한 세 seed 모델에 조건부로, gene을 복원 추출하고 해당 gene의 eligible promoter loci를 함께 재표집하는 bootstrap을 5,000회 수행했다. 이번 test에서는 eligible locus와 gene이 각각 23개로 일대일이었다.

| Within-promoter AUROC 대비 | 평균 차이 | 조건부 95% 구간 |
|---|---:|---:|
| E+P − promoter | +0.1288 | [−0.0200, +0.2709] |
| E+P − enhancer | +0.0387 | [−0.0404, +0.1487] |
| E+거리 − enhancer | +0.1118 | [−0.0166, +0.2518] |
| E+P+거리 − E+P | +0.0997 | [−0.0737, +0.2793] |
| E+P+거리 − E+거리 | +0.0266 | [−0.0459, +0.1014] |
| E+P+거리 − 거리 | +0.0170 | [−0.0582, +0.0964] |
| E+거리 − 거리 | −0.0095 | [−0.1048, +0.0769] |
| E+P+거리 − 4-mer+거리 | +0.0303 | [−0.0302, +0.0971] |

사전에 지정한 대비 8개의 구간이 모두 0을 포함했다. 개선이 없거나 방법이 동등하다는 증거는 아니다. **평가 gene 수가 적고 gene별 결과가 달라, 추가 이득을 확정하기에는 불확실성이 크다.** 재학습 변동, 세 test 염색체 밖의 변동, 다중 비교 보정은 포함하지 않았다.

![Promoter별 성능과 평가 후보 수](C:/Users/public-user/Documents/ChatGPT/FNO/docs/crispr_ranking_results/per_promoter_auroc.png)

Promoter별 후보 수는 2–141개로 크게 다르다. RPN1과 H1FX에는 각각 140개와 141개 후보가 있지만, 후보가 두 개뿐인 promoter도 많다. Macro 지표는 후보 수가 많은 두 유전자에 평가가 지배되지 않도록 각 promoter에 같은 가중치를 부여한다. 그만큼 작은 후보군에서 한 쌍의 순위 변화가 크게 반영될 수 있다. 개별 유전자 그림은 탐색 결과이며 특정 유전자에 대한 기능적 결론으로 해석하지 않는다.

## 데이터 처리와 검증

[연구실 공개 benchmark](https://github.com/EngreitzLab/CRISPR_comparison/tree/50587422e6b11259ead6fbc6f867681c788f39b7/resources/crispr_data)의 고정 commit을 사용했다. 원본 10,356행에는 Nasser2021, Gasperini2019, Schraivogel2020 K562 CRISPR 자료가 통합되어 있다. 원본의 `Regulated`와 `Significant & EffectSize<0`가 일치함을 확인했다.

TSS 좌표가 0인 81행과 enhancer/promoter 입력 window가 겹치는 2행을 제외했다. 총 83행 제외 후 10,273행을 사용했다. 제외 이유와 원본 행 번호를 보존했으며, 10% 초과 N 또는 split 간 exact/RC alias로 인한 추가 제외는 없었다.

| Split | 염색체 | 후보 쌍 | 양성 | 양성·음성이 함께 있는 promoter |
|---|---|---:|---:|---:|
| Train | 아래 dev/test 외 염색체 | 7,468 | 321 | 82 |
| Dev | chr2/4/6 | 924 | 70 | 18 |
| Test | chr1/3/5 | 1,881 | 79 | 23 |

분할 사이의 chromosome, gene ID, promoter locus, enhancer 좌표 및 두 역할을 가로지르는 exact/RC 입력 서열 중복은 0이었다. 가까운 유전체 window도 같은 염색체 안에 있으므로 분할을 넘지 않는다. 다른 염색체 사이의 근접 상동성은 통제하지 않았다.

입력은 enhancer interval midpoint 중심 1kb와 startTSS 중심 1kb다. 전체 perturbed interval의 길이는 다양하므로 이 window가 실제 억제 영역 전체와 항상 일치하지는 않는다. 두 구간 사이의 연속 DNA는 입력하지 않았다. 원본의 거리 값과 재계산 거리가 달라, **실제 GRCh38 interval midpoint–TSS 거리**를 다시 계산해 사용했다.

Ensembl GRCh38의 positive-reference 방향 서열을 조회하고 assembly·좌표·길이를 검사했다. 6개 분산된 window가 UCSC hg38 서열과 정확히 일치했다. 총 6,075개 유일한 window의 응답과 hash를 보존했다. 원본에 strand가 없어 strand에 맞춘 promoter 방향 전환은 하지 않았다. Reference genome이며 K562 개별 유전형이나 구조변이는 반영하지 않는다.

## 학습 조건

기존 pinned DNABERT-2 encoder를 고정하고 각 1kb를 독립 인코딩했다. DNA token만 mean pooling하며 특수 token과 padding을 제외했다. Encoder는 BF16, pooling은 float32다. Test의 유일한 서열은 BPE 160–228 token이었고 truncation은 없었다. 같은 서열은 한 번만 추출해 반복 사용했다.

| 신경망 조건 | 선택 LR | 학습 parameter | Best epoch (42/43/44) |
|---|---:|---:|---|
| Promoter | 0.0001 | 237,464 | 1 / 1 / 1 |
| Enhancer | 0.0003 | 237,464 | 8 / 5 / 5 |
| E+P | 0.0001 | 237,002 | 8 / 9 / 8 |
| E+거리 | 0.0001 | 237,770 | 4 / 5 / 5 |
| E+P+거리 | 0.001 | 237,154 | 8 / 7 / 4 |

조건별 LR 세 개 × 탐색 seed 두 개로 동일하게 탐색하고, 평균 best-dev macro AUROC로 LR을 선택한 뒤 seed 44를 추가했다. 최대 30 epoch, patience 6, batch 64와 train label 비율로 계산한 balanced cross entropy를 공통 사용했다. 최종 모델은 모두 30 epoch 전에 종료했다. Promoter-only는 주요 지표가 늘 0.5라 동점 규칙에 따라 낮은 LR과 첫 epoch가 선택된다. 이 모델의 전체 분류 지표를 최적화한 실험은 아니다.

Logistic 모델은 거리 단독, GC+거리, 4-mer+거리 각각 C 다섯 개를 같은 dev 지표로 선택했다. 모두 C=0.001이 선택되었다. Scaler와 거리 정규화 통계 및 class weight는 train에만 의존한다. Effect size, p-value, power, direct-effect probability, gene ID, dataset ID는 예측 feature로 사용하지 않았다.

Inference에서는 동일한 feature vector를 한 번만 계산하고 확률을 원래 행에 배치했다. Batch shape에 따른 미세한 GPU 반올림 차이가 동일 promoter의 후보 사이에 가짜 순위를 만드는 것을 막았다. 모든 promoter-only 그룹의 AUROC가 정확히 0.5임을 검사했다.

## 비용·재현성과 연구 판단

다운로드·준비를 제외한 학습 및 평가 실행 루프는 **229.8초(약 3.83분)**였다. 특징 추출은 train 19.62초, dev 3.93초, test 5.76초였고, 35회 neural train loop 합은 152.03초였다. 짧은 구간을 독립 인코딩한 pooled-feature 실험이므로 이전 긴 입력 FNO 실험과 직접적인 속도 비교를 하지 않는다.

코드·원본 표·API 응답·데이터·특징·checkpoint·예측 hash를 검사했다. Encoder 가중치는 추출 전후 변하지 않았다. Dev-only checkpoint/LR/C 선택, 좌표에서 재구성한 거리, train-only 정규화, 전체 및 promoter별 지표와 평균/SD를 확인했다. 최종 신경망 15개와 logistic 3개의 확률을 다시 생성해 저장된 예측과 일치함을 검증했다.

이번 실험으로 동일 promoter 내 후보를 평가하는 체계를 만들었고, **거리만으로 얻는 성능을 먼저 넘어야 한다는 사실**을 확인했다. 관측된 서열 추가 이득은 작고 불확실했다. 다음 우선순위는 새 promoter를 더 확보하고, 비슷한 거리에 있는 후보끼리 비교하거나 거리 구간별 성능을 확인하는 것이다. 세포 유형에 맞는 chromatin feature의 추가 이득도 이후 별도 검증 대상으로 둘 수 있다.

현재 23개 promoter의 test로 반복 튜닝하면 개발에 적응할 수 있으므로, 다음 모델 변경에서 이 test를 새로운 확증 평가처럼 취급하면 안 된다. 이번 결과는 sequence-only 정보가 원천적으로 부족하다는 증명도, FNO의 일반적 유용성에 대한 직접 검증도 아니다. FNO로 확장하려면 먼저 거리 대비 안정적인 이득과 더 충분한 평가 표본을 확보하는 것이 타당하다.

## 재현 자료

- [학습 전 고정한 프로토콜](C:/Users/public-user/Documents/ChatGPT/FNO/docs/CRISPR_RANKING_PROTOCOL.md)
- [전체 수치와 bootstrap](C:/Users/public-user/Documents/ChatGPT/FNO/docs/crispr_ranking_results/TABLES.md)
- [최종 18회 평가 CSV](C:/Users/public-user/Documents/ChatGPT/FNO/docs/crispr_ranking_results/all_evaluations.csv)
- [Promoter별 비교 CSV](C:/Users/public-user/Documents/ChatGPT/FNO/docs/crispr_ranking_results/per_promoter_comparison.csv)
- [독립 검증 기록](C:/Users/public-user/Documents/ChatGPT/FNO/docs/crispr_ranking_results/verification.json)
- [Artifact hash 목록](C:/Users/public-user/Documents/ChatGPT/FNO/docs/crispr_ranking_results/export_manifest.json)
- [데이터 준비 스크립트](C:/Users/public-user/Documents/ChatGPT/FNO/scripts/prepare_crispr_ranking.py)
- [실행 설정](C:/Users/public-user/Documents/ChatGPT/FNO/configs/crispr_ranking.json)
- [실행 스크립트](C:/Users/public-user/Documents/ChatGPT/FNO/scripts/run_crispr_ranking.py)
- [검증·보고 스크립트](C:/Users/public-user/Documents/ChatGPT/FNO/scripts/report_crispr_ranking.py)

Study ID: `9c9bcd4d65b1d65e52b59a397eb2c6084439241a67aa73413ef14b9ac89703f4`.

Test JSON SHA-256: `5641c81f088e9952580d5de265221e3693863ad6c6b3cf60076ba8d002d3db5f`.
