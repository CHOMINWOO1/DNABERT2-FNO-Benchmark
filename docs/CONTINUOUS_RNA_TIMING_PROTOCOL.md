# 연속 RNA 실험의 보조 추론 시간 측정

학습 진행 중, test 성능을 보기 전에 정한 시스템 측정이다. 주 학습 프로토콜·설정·모델 선택은 변경하지 않는다. 고정 GPU cache가 모든 길이에서 상주하므로 cache 기반 메모리만으로 실제 서열 입력 비용을 비교할 수 없는 점을 보완한다.

최종 seed 42 checkpoint를 모델 3개 × 길이 3개에서 사용한다. 각 조건에 동일한 dev의 앞 4개 global ID를 쓰며, 선정에 target이나 성능을 사용하지 않는다. RAM에 준비된 실제 DNA에서 BPE tokenization → 독립 1kb DNABERT-2 encoder → 256bp offset-weighted bin pooling → 전역 mixer/head까지 실행한다. 모델·서열 파일을 디스크에서 읽는 초기 시간은 제외하고, 토큰화·CPU/GPU 이동·encoder·pooling·head는 포함한다. 전역 feature cache는 GPU에 올리지 않는다.

지역 encoder batch는 기존과 같은 16이다. 1회 warm-up 후 5회 반복하여 median/min/max 초, gene/초, bp/초와 peak allocated/reserved GPU memory를 기록한다. 4개 gene을 한 호출 묶음으로 처리하는 측정이며 전체 test cohort의 처리 성능을 대표한다고 단정하지 않는다. 테스트 label을 읽거나 성능 지표를 계산하지 않으며 이 결과로 checkpoint를 재선택하지 않는다. Encoder 출력은 batch 구성에 따른 부동소수점 차이가 있을 수 있어 저장 cache와의 완전 일치를 이 시간 측정의 전제로 두지 않는다.
