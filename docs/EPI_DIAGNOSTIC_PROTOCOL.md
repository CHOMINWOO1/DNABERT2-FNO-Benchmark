# 5kb EPI 구조·pooling 진단 실험

이번 질문은 FNO의 긴 입력 결과가 제한된 구조·학습률 설정이나 평균 pooling과 관련되는가이다. 기존 5kb train 1,024 / dev 256쌍과 DNABERT-2 frozen feature cache를 재사용한다. 기존 test 256쌍은 이미 본 탐색 자료다.

## 사전에 고정한 탐색과 평가

FNO는 modes 8/16/64, CNN은 circular kernel 3/9/17을 탐색한다. 각 구조에서 폭을 조절해 학습 parameter를 237,571개 기준 1% 이내로 맞춘다. 이는 mode만의 인과 효과가 아니라, 일정 용량 안에서 mode와 폭을 함께 바꾸는 구조 탐색이다. 두 방법 모두 학습률 0.0001/0.0003/0.001과 seed 42/43을 사용한다. 방법당 9개 설정 × 2개 seed, 총 36회다.

각 방법에서 두 seed의 best-dev MCC 평균으로 설정 하나를 선택한다. 동점이면 작은 LR, 다음에는 작은 mode/kernel 값을 우선한다. Test는 선택에 사용하지 않는다. 최대 12 epochs, patience 4, min_delta 0.0001, cosine LR, batch 2 × gradient accumulation 16, BF16, 원본 encoder 고정은 이전 실험과 같다. 탐색 횟수와 최대 update 수는 같지만 조기 종료 시점 및 실제 계산 시간은 다를 수 있다.

선택된 구조·LR에서 mean pooling seed 44를 추가하고, attention pooling을 seed 42/43/44로 실행한다. 기존 mean run을 재사용하므로 추가 8회, 총 학습 횟수는 44회다. 최종 비교는 FNO/CNN × mean/attention × 3개 seed의 12개 모델이다. Pooling 조건마다 별도 구조·LR 최적화를 하지 않으며, 평균 pooling에서 선택한 설정에 조건부인 비교다.

Attention pooling은 adapter 출력의 각 DNA token에 하나의 학습 가능한 선형 점수를 주고, mask를 적용한 softmax 가중 평균을 계산한다. 점수 weight를 0으로 초기화하여 처음에는 기존 평균 pooling과 같은 값을 내게 한다. 추가 parameter는 768개다. 같은 seed의 classifier head와 adapter 초기값을 맞추며, dropout 및 원본 encoder 처리 규칙은 같다. Padding 무시, 초기 출력 일치, scorer의 유효 gradient를 단위 테스트로 확인했다.

## 새로운 평가 cohort

`data/epi_diagnostic_holdout/`에 1,024쌍(label별 512)을 seed 20260911로 고정했다. 원본 GM12878 14,000쌍에서 이전 train/dev/test 전체와 enhancer 또는 promoter component 그룹을 공유하는 후보를 제외했다. 그룹은 이전 1/2/5kb 모든 중심 crop의 exact/RC 별칭을 연결하여 만든다. 남은 후보는 label 0: 5,830쌍, label 1: 5,489쌍이다.

새 test 파일 SHA256은 `377dca3e32746ea414a986e4baff31eb31aceeecbedebc815004e9b186bd1b79`이다. 모델 탐색 전 파일과 원본 매핑을 고정했다. 이전 모든 split과 enhancer/promoter 그룹 겹침은 각각 0이다. 이 cohort는 같은 GM12878 공개 자료에서 만든 미사용 평가 표본이며 외부 데이터셋 검증은 아니다. Near-homology, 역할 간 유전체 overlap, chromosome 분리는 통제하지 못했다.

전체 44회 학습과 모델 선택이 끝난 다음 최종 12개 모델을 기존 test와 새 cohort에서 평가한다. 이전 5kb CNN/FNO/LoRA의 3개 seed checkpoint도 변경 없이 새 cohort에서 평가해 참고 기준으로 둔다. LoRA를 이번 탐색 예산으로 새로 최적화하지 않으므로 최적화된 LoRA와의 최종 우열 비교로 해석하지 않는다. 모든 평가에서 MCC, Accuracy, F1, ROC-AUC, PR-AUC, AP와 row별 확률을 보관한다.

새 cohort에서 FNO−CNN을 pooling별로, attention−mean을 방법별로 비교한다. 기술적 불확실성은 동일 test row를 label별로 함께 재표집하는 paired bootstrap 5,000회(seed 20260912)로 구한다. 3개 고정 학습 모델의 평균 MCC 차이를 사용한다. 재학습 변동·공유 component 간 상관·염색체 불확실성·다중 비교 보정은 포함하지 않는다. 새 holdout 결과를 보고 설정을 다시 고르거나 재학습하지 않는다.

## 재현

기존 긴 입력 실험의 데이터·모델·캐시가 준비된 환경에서 실행한다. 첫 데이터 준비 명령은 새 cohort가 없을 때만 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_diagnostic_holdout.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_epi_diagnostic.py
```

학습 중단 후 같은 runner로 재개할 수 있다. 설정·source·환경·holdout identity가 바뀌면 기존 output으로 재개하지 않는다. 기존 `dnabert_fno` 소스와 이전 결과는 변경하지 않았다. 새 runner에서만 classifier factory를 교체하여 검증된 학습 loop를 재사용한다. 프로토콜은 `runs/epi_diagnostic/protocol.json`, 진행 상태는 `runs/epi_diagnostic/status.json`에 기록한다.
