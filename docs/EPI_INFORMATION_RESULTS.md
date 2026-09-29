# EPI 입력 정보 진단 결과

2026-09-09. **Promoter만 사용한 DNABERT-2 모델이 이번 조건 중 평균 MCC와 ROC-AUC가 가장 높았다.** 두 요소를 각각 인코딩하여 결합해도 promoter 단독 대비 추가 이득을 확인하지 못했다. 두 서열을 연결하여 인코딩한 모델은 각각 인코딩한 모델보다 낮았다. 따라서 지금은 FNO를 더 튜닝하기보다, 두 요소의 관계를 평가할 수 있는 데이터 설정과 각 요소의 정보를 보존하는 입력 방식을 우선 검토하는 것이 타당하다.

RTX 5060 Ti 8GB에서 신경망 **35회 학습**, GC·4-mer logistic regression **30회 fit**, 최종 **21회 평가**를 완료했다. 모든 선택을 dev에서 끝낸 뒤 새 holdout 1,024쌍을 평가했다. 전체 실행 루프는 198.7초(약 3.31분)였다. 코드 테스트 31개 및 저장된 모든 최종 예측의 재생성·지표 검증을 통과했다.

## 같은 새 holdout에서의 결과

신경망 결과는 seed 42/43/44의 평균 ± 표본 SD이다. GC·4-mer는 deterministic logistic regression 한 모델의 점수이므로 seed SD가 없다. 모든 행은 동일한 양성 512쌍·음성 512쌍에서 계산했다.

| 입력과 모델 | MCC | ROC-AUC |
|---|---:|---:|
| DNABERT-2: enhancer 단독 | 0.1296 ± 0.0097 | 0.5840 ± 0.0048 |
| DNABERT-2: promoter 단독 | **0.2330 ± 0.0295** | **0.6694 ± 0.0157** |
| DNABERT-2: 각각 인코딩 후 벡터 결합 | 0.2043 ± 0.0384 | 0.6618 ± 0.0142 |
| DNABERT-2: 연결 인코딩 후 영역별 pooling | 0.1310 ± 0.0340 | 0.6062 ± 0.0109 |
| DNABERT-2: 연결 인코딩 후 전체 pooling | 0.1585 ± 0.0171 | 0.5858 ± 0.0037 |
| GC: enhancer 단독 | 0.0685 | 0.5396 |
| GC: promoter 단독 | 0.0820 | 0.5618 |
| GC: 두 요소의 특징 결합 | 0.0899 | 0.5597 |
| 4-mer: enhancer 단독 | 0.1192 | 0.5851 |
| 4-mer: promoter 단독 | 0.0977 | 0.5756 |
| 4-mer: 두 요소의 특징 결합 | 0.1446 | 0.6097 |

![EPI 입력 정보 진단의 MCC와 ROC-AUC](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_information_results/information_comparison.png)

**Promoter 단독 모델은 두 요소를 함께 관찰하지 않아도 상당한 분류 신호를 얻었다.** 개별 인코딩 후 결합의 평균 MCC는 enhancer 단독보다 높지만 promoter 단독보다 낮다. 이는 결합 모델의 점수에서 promoter 쪽 정보가 큰 역할을 할 수 있음을 시사한다. 그러나 각 특징의 인과적 기여도를 분해한 실험은 아니다.

**연결 인코딩의 추가 이득은 확인하지 못했다.** 영역별 pooling을 유지한 동일 차원의 비교에서, 연결 인코딩은 별도 인코딩 후 결합보다 세 seed 모두 낮았다. 전체 pooling도 promoter 단독보다 수치상 낮았다. 연결했을 때 긴 입력, 위치 변화, 인공적 경계, tokenization 변화가 함께 생기므로 어느 요인이 원인인지는 이번 실험으로 분리되지 않는다.

GC·4-mer도 일부 분류 신호를 담지만 promoter DNABERT-2의 성능을 그대로 재현하지는 못했다. 예를 들어 promoter 단독의 ROC-AUC는 DNABERT-2 0.6694, GC 0.5618, 4-mer 0.5756이었다. 이 수치만으로 조성 신호가 전부를 설명한다거나 데이터 누출이 발생했다고 결론 내릴 수 없다.

## 차이의 불확실성

사전에 지정한 대비에 대해 label-stratified paired row bootstrap 5,000회를 수행했다. 같은 행을 두 모델에 함께 재표집하고, 신경망은 각 seed의 MCC를 계산한 뒤 평균했다. 아래 구간은 고정된 학습 모델에 조건부이며 ensemble MCC의 구간이 아니다.

| MCC 대비 | 평균 차이 | 조건부 95% 구간 |
|---|---:|---:|
| 각각 인코딩 후 결합 − enhancer 단독 | +0.0747 | [+0.0223, +0.1261] |
| 각각 인코딩 후 결합 − promoter 단독 | −0.0288 | [−0.0769, +0.0177] |
| 연결 인코딩·영역 pooling − 각각 인코딩 후 결합 | −0.0732 | [−0.1138, −0.0312] |
| 연결 인코딩·영역 pooling − promoter 단독 | −0.1020 | [−0.1539, −0.0502] |
| 연결 인코딩·영역 pooling − 연결 인코딩·전체 pooling | −0.0275 | [−0.0656, +0.0137] |

Promoter 단독과 별도 인코딩 후 결합의 차이는 구간이 0을 포함한다. **Promoter 단독이 확정적으로 우월하거나 두 모델이 동등하다고 주장하지 않는다.** 현재 예산에서 결합의 추가 이득을 확인하지 못한 것이다. 연결 인코딩의 열세는 위 조건부 구간에서 0을 포함하지 않지만, 여러 대비에 대한 다중 비교 보정이나 재학습 변동을 반영한 확증적 검정은 아니다. 사전 지정 대비 10개의 전체 결과는 [수치표](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_information_results/TABLES.md)에 보존했다.

## 비교 조건과 데이터 검증

기존 EPI train 1,024쌍과 dev 256쌍을 순서까지 유지했다. 새 test는 이전 EPI train/dev/test 전체와 지난 추가 진단 holdout 1,024쌍의 enhancer·promoter 구성 서열 그룹을 제외한 뒤 추출했다. 이전 1/2/5kb crop의 exact/RC 일치와 전이적 연결을 모두 반영한 그룹 중복이 두 역할 모두 0이었다.

새 test에는 enhancer 그룹 1,023개, promoter 그룹 993개가 있다. 일부 행이 구성 서열을 공유하므로 독립 표본이라는 가정에 한계가 있다. Bootstrap은 이 상관을 조정하지 않았다. 이 holdout은 같은 GM12878 corpus 내부의 미사용 표본이며, 외부 데이터셋이나 염색체 분리 검증이 아니다. 근접 상동성과 역할 간 유전체 중복은 통제하지 않았다. 향후 모델 변경에서는 이번 test도 이미 관찰한 개발 자료로 취급해야 한다.

Frozen DNABERT-2의 고정 revision을 사용해 enhancer 3,000bp와 promoter 2,000bp를 인코딩했다. 이 5kb 연결 입력은 두 요소 사이의 실제 유전체 구간을 포함하지 않는다. Fast tokenizer의 bp offset으로 영역별 pooling 경계를 정하고, 경계에 걸친 BPE token은 겹치는 bp 비율에 따라 두 영역에 나눠 반영했다. Padding과 특수 token은 제외했으며, 모든 입력에서 truncation은 0이었다. BPE 표현을 bp 균일 격자로 재표집하는 실험은 아직 하지 않았다.

분류기는 LayerNorm → Linear → GELU → Dropout → Linear이다. 단독/전체 pooling 입력은 768차원, pair pooling 입력은 1,536차원이다. 학습 parameter는 각각 **237,464개와 237,002개**로 맞췄다. Pair 비교는 같은 구조와 seed별 동일 초기값을 사용한다. 단독 대 pair는 입력 차원과 hidden width가 다르므로 capacity까지 완전히 동일하다는 주장은 하지 않는다. Separate pair의 nonlinear 분류기도 두 벡터의 상호작용을 학습할 수 있다.

| 조건 | Dev로 선택한 LR | Best epoch (seed 42/43/44) |
|---|---:|---|
| Enhancer 단독 | 0.0003 | 4 / 6 / 2 |
| Promoter 단독 | 0.001 | 20 / 12 / 9 |
| 각각 인코딩 후 결합 | 0.0003 | 8 / 24 / 11 |
| 연결 인코딩·영역 pooling | 0.001 | 9 / 8 / 4 |
| 연결 인코딩·전체 pooling | 0.001 | 2 / 6 / 2 |

조건마다 LR 세 개 × 탐색 seed 두 개의 동일 예산을 사용한 뒤, 선택 LR에 seed 44를 추가했다. 공통 최대 30 epoch, patience 6, batch 32를 적용했다. 별도 인코딩 후 결합의 seed 43은 최대 30 epoch까지 실행했고 나머지 최종 모델은 그전에 종료했다. 수렴한 최적 성능을 입증한 실험은 아니다. GC·4-mer의 scaler는 train에만 fit했고 C 다섯 개 중 dev MCC로 선택했다. Threshold는 test에서 조절하지 않고 모두 0.5로 고정했다.

이번 분류기는 이전 CNN/FNO head와 다르고 학습 예산 및 평가 표본도 바뀌었다. 따라서 이전 FNO 점수와 이번 promoter 점수의 차이를 같은 조건에서 얻은 모델 개선량으로 해석할 수 없다.

## 비용·재현성과 다음 판단

Encoder 특징 추출은 train 57.00초, dev 14.37초, test 56.97초였다. 전체 35회 head 학습의 train loop 합은 43.75초였다. Pooling한 작은 벡터를 저장해 학습하므로 token별 CNN/FNO 연산을 수행했던 이전 실험보다 학습 문제가 단순하다. 실행 시간 차이를 FNO와의 알고리즘 효율 비교로 해석하지 않는다.

모든 split의 encoder parameter hash가 추출 전후 동일했다. 코드·데이터·특징·checkpoint·예측 hash와 원본 서열·label 복원을 검증했다. Dev의 best epoch와 hyperparameter 선택을 다시 계산했고, 최종 신경망 15개 및 logistic 모델 6개의 확률을 checkpoint에서 재생성하여 저장된 예측과 일치함을 확인했다. 모든 지표와 평균·SD도 별도로 재계산했다.

**현재 가장 중요한 후속 질문은 “두 요소의 관계를 요구하는 평가를 어떻게 만들 것인가”다.** 다음 데이터에서는 promoter 단독 모델을 필수 기준으로 두고, 같은 promoter에 대응하는 여러 enhancer 후보와 실제 거리·좌표를 활용해 결합 정보의 추가 이득을 평가하는 설계가 우선이다. 이런 메타데이터를 확보한 뒤 염색체 또는 영역을 분리해 검증해야 한다. 실제 장거리 의존성을 다루려면 연속 구간 또는 검증된 조절 관계를 포함하는 평가가 필요하다.

이번 조건에서는 개별 인코딩 후 결합을 연결 입력보다 우선 검토할 근거가 있다. 그러나 이를 더 복잡한 FNO 결합으로 확장하기 전에 promoter 단독 이상의 안정적인 이득을 보여야 한다. 현재 결과는 생물학적 enhancer–promoter 상호작용이 없다는 결론이나 FNO의 일반적 한계를 입증하지 않는다.

## 재현 자료

- [학습 전에 고정한 프로토콜](C:/Users/public-user/Documents/ChatGPT/FNO/docs/EPI_INFORMATION_PROTOCOL.md)
- [전체 수치·bootstrap 대비](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_information_results/TABLES.md)
- [21회 평가 CSV](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_information_results/all_evaluations.csv)
- [검증 기록](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_information_results/verification.json)
- [원본 artifact hash 목록](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_information_results/export_manifest.json)
- [실험 설정](C:/Users/public-user/Documents/ChatGPT/FNO/configs/epi_information.json)
- [실행 스크립트](C:/Users/public-user/Documents/ChatGPT/FNO/scripts/run_epi_information.py)
- [검증·분석 스크립트](C:/Users/public-user/Documents/ChatGPT/FNO/scripts/report_epi_information.py)

Study ID: `66591f94a7c3c2e0f97d114f79f4cf106dbe9fd3cd0154402cdc60a2e8f2c00b`.

새 test CSV SHA-256: `75e8605a41d4ba2c66f6f912363c6a4346afd6634ad64a23fb337811e8cd968d`.
