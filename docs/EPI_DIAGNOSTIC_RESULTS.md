# 5kb EPI 추가 실험 결과

2026-09-09. RTX 5060 Ti 8GB에서 **44회 학습과 33회 최종 평가**를 완료했다. 실행 루프는 약 **54.04분** 걸렸다. 구조·학습률 튜닝과 attention pooling으로 FNO의 새 holdout MCC가 조금 상승했지만, **같은 조건의 CNN보다 FNO가 확실히 좋다는 근거는 얻지 못했다.** 현재의 frozen DNABERT-2 뒤에 FNO를 붙이는 설정을 더 튜닝하는 우선순위는 낮추는 것이 타당하다.

아래는 모두 **같은 새 holdout 1,024쌍**에서 계산한 값이다. ±는 seed 42/43/44의 평균에 대한 **표본 표준편차**이며 신뢰구간이 아니다. ‘기존’ 모델은 이전 5kb 실험의 checkpoint를 그대로 새 holdout에 평가했다.

| 모델 | MCC (평균 ± SD) | ROC-AUC (평균 ± SD) |
|---|---:|---:|
| 기존 CNN | 0.1156 ± 0.0116 | 0.5923 ± 0.0027 |
| 튜닝 CNN + mean | 0.1247 ± 0.0163 | 0.5959 ± 0.0044 |
| 튜닝 CNN + attention | 0.1266 ± 0.0147 | 0.5956 ± 0.0043 |
| 기존 FNO | 0.1073 ± 0.0184 | 0.5956 ± 0.0028 |
| 튜닝 FNO + mean | 0.1236 ± 0.0102 | 0.5864 ± 0.0075 |
| 튜닝 FNO + attention | 0.1371 ± 0.0200 | 0.5883 ± 0.0080 |
| 기존 LoRA (추가 튜닝 없음) | 0.1480 ± 0.0568 | 0.6107 ± 0.0439 |

![새 holdout의 모델별 MCC와 ROC-AUC](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_diagnostic_results/holdout_comparison.png)

FNO의 MCC는 튜닝으로 **+0.0163**, attention pooling으로 추가 **+0.0135** 상승했다. 그러나 attention의 이득은 3개 seed 중 2개에서만 나타났고, seed별 차이는 +0.0042, +0.0422, −0.0059로 seed 43의 영향이 컸다. FNO의 ROC-AUC는 기존 0.5956에서 튜닝 후 0.5864, attention 추가 후 0.5883으로 상승하지 않았다. 따라서 MCC 상승만으로 전반적인 예측 순위 성능이 좋아졌다고 해석할 수 없다.

LoRA는 평균 MCC와 ROC-AUC가 가장 높지만 seed 간 편차가 크고 이번에 추가 튜닝하지 않았다. 이 표는 최적화된 LoRA와의 비교나 LoRA의 확정적 우위를 뜻하지 않는다. 이전 256쌍 test의 점수와 이번 1,024쌍 holdout 점수는 평가 표본이 다르므로 직접적인 전후 개선 비교에 사용하지 않는다.

## 비교가 얼마나 확실한가

새 holdout에서 같은 행을 함께 재표집하는 label-stratified paired bootstrap을 5,000회 수행했다. 각 재표집에서 세 seed 모델의 MCC를 각각 계산한 뒤 평균했다. 앙상블 예측의 MCC가 아니다.

| MCC 차이 | 관측 평균 차이 | 조건부 95% bootstrap 구간 |
|---|---:|---:|
| FNO mean − CNN mean | −0.0011 | [−0.0282, +0.0262] |
| FNO attention − CNN attention | +0.0105 | [−0.0098, +0.0316] |
| FNO attention − FNO mean | +0.0135 | [−0.0068, +0.0336] |
| CNN attention − CNN mean | +0.0019 | [−0.0041, +0.0080] |
| FNO의 pooling 이득 − CNN의 pooling 이득 | +0.0116 | [−0.0093, +0.0330] |

모든 구간이 0을 포함한다. FNO 특유의 이득이나 attention pooling의 안정적인 개선을 확정할 수 없으며, 두 방법이 동등하다는 증거도 아니다. 이 구간은 이미 학습된 세 모델에 조건부인 행 단위 불확실성이다. 재학습의 불확실성, 같은 구성 서열을 공유하는 holdout 행 간 상관, 염색체별 변동은 반영하지 않았고 다중 비교 보정도 하지 않았다.

## 무엇을 바꾸고 통제했나

이전 5kb EPI의 train 1,024쌍과 dev 256쌍을 유지했다. Frozen DNABERT-2의 캐시된 hidden states를 사용하고, batch 2 / accumulation 16, 최대 12 epoch, early stopping patience 4, cosine 학습률, dropout 0.1을 동일하게 적용했다. DNABERT-2 encoder의 원래 가중치는 고정했다.

| 항목 | FNO | CNN |
|---|---|---|
| 구조 후보 | modes 8/16/64 | circular kernel 3/9/17 |
| 대응 width | 80/64/37 | 116/94/79 |
| 학습률 후보 | 0.0001 / 0.0003 / 0.001 | 0.0001 / 0.0003 / 0.001 |
| 탐색 예산 | 9개 설정 × 2 seed | 9개 설정 × 2 seed |
| dev로 선택한 설정 | modes 64, width 37, LR 0.0003 | kernel 17, width 79, LR 0.0003 |
| 선택 설정의 평균 best-dev MCC | 0.1828 | 0.1898 |
| 선택 mean 모델의 학습 parameter | 237,424 | 237,920 |

Mean pooling 후보의 parameter 수는 목표 237,571의 1% 이내로 맞췄다. Modes/kernel과 width가 함께 바뀌므로 순수한 modes 단독 ablation은 아니다. seed 42/43의 평균 best-dev MCC로 설정을 선택한 뒤 mean seed 44를 추가했다. 선택한 각 설정에 attention pooling을 적용해 seed 42/43/44를 학습했다. 총 36회 탐색 + 2회 mean 추가 + 6회 attention = 44회다.

Attention pooling은 DNA token별 학습 가능한 선형 점수와 masked softmax이며, 768개 parameter를 추가한다. 점수 가중치를 0으로 초기화해 최초 pooling은 masked mean과 같게 했다. PAD/CLS/SEP는 제외한다. 같은 seed에서 adapter와 classifier 초기값 및 학습 RNG를 맞췄다. Attention에는 mean으로 선택한 구조와 학습률을 그대로 사용했으며, 별도로 최적화하지 않았다.

![Dev에서만 수행한 구조와 학습률 탐색](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_diagnostic_results/dev_search.png)

44회 학습을 모두 마친 뒤 최종 12개 모델을 기존 test와 새 holdout에 평가했다(24회). 기존 CNN/FNO/LoRA의 seed별 checkpoint 9개를 새 holdout에 추가 평가해 총 33회다. LoRA 평가는 병합하지 않은 adapter를 사용했다. 최종 선택 모델의 best epoch는 1–5였으며 모두 early stopping으로 종료했다. 충분히 수렴한 최적 성능이라는 주장은 하지 않는다.

## 새 holdout과 적용 범위

같은 공개 GM12878 EPI 원본에서 양성 512쌍, 음성 512쌍을 추출했다. 이전 EPI train/dev/test 전체 1,536쌍과 enhancer 또는 promoter 구성 서열 그룹이 겹치는 후보를 제외했다. 그룹은 이전 1/2/5kb crop들의 exact/reverse-complement 일치와 전이적 연결을 반영했다. 원본 서열·label 재구성과 그룹 비중복 검증을 통과했다.

새 holdout은 **같은 데이터셋 안의 미사용 표본**이다. 외부 데이터셋 검증이나 염색체 holdout은 아니다. 근접 상동성, enhancer/promoter 역할을 가로지르는 유전체 중복, 염색체 분리는 통제하지 않았다. Holdout 내부에서 구성 서열을 공유할 수도 있다.

입력 5kb는 enhancer 3kb와 promoter 2kb를 이어 붙인 쌍이다. 연속된 유전체 5kb 영역이나 두 요소 사이의 실제 genomic distance를 입력한 실험이 아니다. 새 holdout의 BPE token 길이는 912–1,100이고 잘림은 없었다. 이번 결과만으로 실제 장거리 조절 관계를 학습했는지 판단할 수 없다.

## 비용과 검증

튜닝 후 선택된 mean/attention head의 평균 학습 시간은 CNN 43.3/43.8초, FNO 42.7/41.8초였다. 이는 캐시 이후 개별 학습 기록의 시간이며, 전체 탐색과 특징 추출 비용을 포함하지 않는다. 실행 순서, FFT 초기화, early stopping이 달라 이전 FNO의 85.7초와 비교해 구조 변경으로 두 배 빨라졌다고 주장할 수 없다. 전체 실행 루프의 54.04분은 탐색·새 holdout 특징 추출·평가를 포함한다.

모듈 테스트 **28개가 통과**했다. 코드·데이터·checkpoint hash, dev 선택, frozen 가중치 유지, 저장된 모든 예측에서 재계산한 지표 및 평균/SD를 검증했다. 기존 기본 설정을 재실행한 FNO/CNN × seed 42/43의 학습 loss와 dev MCC trajectory는 이전 실행과 차이가 정확히 0이었다.

이번 추가 실험은 튜닝 부족과 pooling 선택이 결과를 설명하는지 확인하는 데 의미가 있었다. 관측된 개선은 작고 지표·seed 간 일관성이 부족했다. 이 설정에서 추가적인 modes·학습률 탐색을 크게 늘릴 근거는 약하다. 이는 다른 task, backbone 내부 교체, 실제 장거리 입력에서 FNO가 효과 없다는 결론으로 확장되지 않는다.

## 재현 자료

- [사전에 고정한 실험 프로토콜](C:/Users/public-user/Documents/ChatGPT/FNO/docs/EPI_DIAGNOSTIC_PROTOCOL.md)
- [전체 수치·탐색표](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_diagnostic_results/TABLES.md)
- [33회 평가 기록 CSV](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_diagnostic_results/all_evaluations.csv)
- [독립 재검증 기록](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_diagnostic_results/verification.json)
- [원본 artifact 복사본의 hash 목록](C:/Users/public-user/Documents/ChatGPT/FNO/docs/epi_diagnostic_results/export_manifest.json)
- [실행 설정](C:/Users/public-user/Documents/ChatGPT/FNO/configs/epi_diagnostic.json)
- [분석·보고 스크립트](C:/Users/public-user/Documents/ChatGPT/FNO/scripts/report_epi_diagnostic.py)
- [이전 1/2/5kb 실험 결과](C:/Users/public-user/Documents/ChatGPT/FNO/docs/LONG_EPI_RESULTS.md)

Study ID: `27b1aa2e9de6d4842e028a1712f4134c122c1c6cec7a59dec8191c051511b922`.

새 holdout CSV SHA-256: `377dca3e32746ea414a986e4baff31eb31aceeecbedebc815004e9b186bd1b79`.
