# DNABERT2-FNO-Benchmark

**Controlled comparisons of Fourier neural operators, convolution, attention, and LoRA for genomic prediction.**

DNABERT-2에 FNO를 결합하고, 비교군·입력 길이·학습 비용을 통제해 실제 이득을 검증한 연구 프로젝트입니다.

## Research question

Can a Fourier operator improve the use of frozen DNA language-model representations, and does that improvement survive stronger baselines and longer-context evaluations?

## What is implemented

- Frozen DNABERT-2 feature extraction, FNO/CNN/MLP heads, LoRA and combined adaptation.
- Promoter classification, enhancer–promoter comparisons, CRISPR candidate ranking, and 4/16/64 kb expression prediction.
- Explicit data splits, multiple seeds, matched baselines, uncertainty estimates, and runtime accounting.

## Selected historical results

| Evaluation | Result | Interpretation |
|---|---|---|
| Promoter comparison, mean test MCC | FNO **0.4805 ± 0.0197**; LoRA **0.6014 ± 0.0223**; LoRA+FNO **0.6059 ± 0.0169** | LoRA outperformed FNO in all five seeds; the combination did not improve in every seed. |
| Continuous DNA, FNO macro Pearson | **0.7026 at 4 kb → 0.7172 at 64 kb** | Exploratory longer-context gain; no established advantage over the 64 kb CNN or attention baselines. |

These are recorded experiment results, not measurements rerun for this GitHub snapshot.

![Continuous-context comparison](docs/continuous_rna_results/continuous_comparison.png)

## Quick start

Python 3.12 is recommended for the recorded environment.

```bash
python -m venv .venv
# Activate .venv using your operating system's command.
python -m pip install -e ".[test]"
python -m pytest tests -q
```

The tests exercise model components without downloading pretrained weights. For full experiments, obtain the model/data through the preparation scripts and follow the recorded protocol; weights, raw data, and training checkpoints are not bundled.

## Explore

- [Comparative protocol](docs/COMPARATIVE_PROTOCOL.md) and [results](docs/COMPARATIVE_RESULTS.md)
- [Continuous-context protocol](docs/CONTINUOUS_RNA_PROTOCOL.md) and [results](docs/CONTINUOUS_RNA_RESULTS.md)
- [CRISPR ranking results](docs/CRISPR_RANKING_RESULTS.md)
- `dnabert_fno/`: model, data, and training components
- `configs/`, `scripts/`, `tests/`: experiment definitions, runners, and regression tests

## Scope

The experiments do not establish a general superiority of FNO over attention, CNNs, or LoRA. Some results are exploratory, depend on restricted datasets or chromosome splits, and use unadjusted uncertainty intervals; consult the individual result documents.

## Publication and validation

This is a curated research source snapshot, not the complete local experiment archive.
See [validation](VALIDATION.md), [publication scope](PUBLICATION_NOTES.md), and [credential handling](SECURITY.md).
