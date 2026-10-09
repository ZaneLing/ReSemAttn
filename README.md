# ReSemReason / ReSemAttn

Implementation of **Beyond Reachability: A Relation-Semantic Reasoning Framework for Biomedical Multi-hop QA**, prepared for the WWW 2027 Semantic and Knowledge track.

ReSemReason shares question-only typed, directed relation schemas between candidate discovery and evidence attention. A correct endpoint can still have invalid support: **reachable-but-invalid**. The returned answer and its selected path are evaluated separately.

## What this release implements

- Full-inventory uncertain grounding with temperature 0.1 and top-N anchors.
- Question-only factorized top-M schema prediction, retained-hypothesis entropy confidence, and bounded schema-guided beam search.
- Entity-type, directed-relation, path and topology memories; nonnegative mismatch energies and schema-marginalized compatibility.
- Two ReSemAttn layers with separate low-rank K/V mixtures, a confidence-weighted prior, and gated residual fusion.
- Candidate-conditioned backbone states, normalized log-mean-exp absolute support, and `argmax(u)` explanations.
- **Full-forward training** through those same attention layers, training-only grounding/schema NLLs, and supplied matched candidate/path pairs.
- Independent sigmoid set decoding, development-only threshold selection, benchmark output adapters, frozen-pool controls, and auditable evidence metrics.

This supersedes the attention-free surrogate trainer in the original repository commit. The manuscript's Appendix C.2 describes that **historical package**; Appendix C's completion requirements are implemented here. See [paper-to-code alignment](docs/PAPER_ALIGNMENT.md) and [protocols and data formats](docs/PROTOCOLS.md).

**Experimental status:** included data are toy fixtures. Formal benchmark datasets, original split manifests, historical checkpoints, independent biomedical adjudications, and per-question records behind the paper's tables are not available in this repository. No table value or red simulated E1–E6 value is used as an evaluation output. Implementing the method does not establish reproduction of its reported scores. The diagram-derived imatinib fixture is explicitly a toy graph, not a verified PrimeKG export.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

For the recorded Llama environment, use Python 3.10.14, the appropriate CUDA 12.1 PyTorch build, `requirements-paper.txt`, and `pip install -e .`. `pip install -e '.[hf]'` installs compatible optional Hugging Face dependencies without pinning the historical environment. Access to Llama-3.1-8B-Instruct weights must be obtained separately.

## Offline smoke run

```bash
python scripts/train.py --config configs/default.yaml --epochs 1 --output outputs/toy.pt
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/toy.pt --output outputs/toy_eval
python scripts/run_inference.py --config configs/default.yaml --checkpoint outputs/toy.pt \
  --question "Which adverse effects are linked to drugs used to treat rheumatoid arthritis?"
```

An **untrained** plumbing demo must opt in explicitly:

```bash
python scripts/run_inference.py --config configs/case_demo.yaml --allow-untrained-toy --toy-oracle \
  --question "Which diseases are associated with proteins targeted by imatinib?"
```

`--toy-oracle` is rejected for benchmark configurations. Untrained toy outputs are labeled accordingly. Benchmark evaluation requires a checkpoint. Missing validity annotations produce `null`, not synthetic or schema-derived measurements.

## Paper configuration

`configs/paper.yaml` and `configs/llama31_8b.yaml` encode the reported settings:

| Setting | Value |
|---|---:|
| Backbone | Llama-3.1-8B-Instruct |
| Anchors N / schemas M / beam B / paths K | 8 / 4 / 32 / 8 |
| Maximum hops | 3 |
| Evidence-attention layers | 2 |
| Relation bases / rank | 8 / 16 |
| Absolute-support weight | 0.35 |
| Initial recorded set threshold | 0.43 |
| Learning rate / effective batch / epochs | 2e-5 / 8 / 5 |
| Seeds | 13, 21, 42, 87, 100 |

The threshold is **refit on development data** on the 0.05–0.95 grid, stored in the checkpoint, and restored for test evaluation. The reported 0.43 is an initial historical setting, not asserted to be the optimum of that grid. Effective batches accumulate per-question gradients before an optimizer update. Gradient checkpointing is available for the unfrozen backbone. The included trainer is a single-process reference implementation; the original four-GPU sharding/optimizer-offload launcher was not supplied and is not claimed to have been reconstructed or hardware-validated.

Populate the frozen KG and the train/dev/test files referenced by the config, then run:

```bash
python scripts/run_seeds.py --config configs/paper.yaml --validity-labels data/biohopr/final_labels.jsonl
```

Use `configs/primekgqa.yaml` or `configs/medreason.yaml` for their dataset paths. Preserve benchmark IDs, source grouping, graph version, preprocessing manifests and file checksums. Do not substitute the toy fixtures or recreate historical split counts by random partitioning.

## Controls and evaluation

- **E1:** `scripts/frozen_pool.py` exports candidate/path pools once and scores their exact bytes with independently trained CE or ReSemAttn checkpoints.
- **E2:** `configs/controls.yaml` defines textual-memory, final-bias, prior-only and KV-only interventions. Train with `--variant NAME`; score with the same variant and frozen pools.
- **E3:** hard-schema training and question-only component/coverage/calibration diagnostics are available. `e3_uniform_schemas` applies uniform retained-schema probabilities. Deliberately wrong-schema cohorts require curated inputs; no such cohort is fabricated.
- **E4:** `scripts/prepare_data.py --mode composition` makes disjoint directed-relation-chain groups; source/template splits are also available. Audit primitive coverage before calling a split “unseen compositions of known relations.”
- **E5:** train `--variant e5_no_absolute_support`; compare on the same normal, all-invalid and singleton pools. Special pools must come from independently labeled paths.
- **E6:** `scripts/annotations.py` exports blinded paths and reconciles two annotators without converting disagreements into negatives. `scripts/analyze_predictions.py` evaluates prespecified strata.

```bash
python scripts/frozen_pool.py export --config configs/paper.yaml --checkpoint outputs/full.pt --pool outputs/pool.jsonl
python scripts/frozen_pool.py score --config configs/paper.yaml --checkpoint outputs/full.pt --pool outputs/pool.jsonl
python scripts/score_path_pairs.py --config configs/paper.yaml --checkpoint outputs/full.pt --questions data/biohopr/path_pairs.jsonl
python scripts/profile_runtime.py --config configs/paper.yaml --checkpoint outputs/full.pt
```

The CE comparator is a trainable question/candidate/path cross-encoder, **not a heuristic score substituted for a trained baseline**. This executable baseline protocol is not evidence that its weights or scores reproduce the original table. All controls need matched data, tuning budgets, seeds and independently selected dev thresholds.

Evaluation writes per-question JSONL, graph/data/checkpoint hashes, global answer metrics, a common-cohort evidence contingency, schema diagnostics, and available benchmark-specific metrics. PairAcc uses strict wins; ties receive zero credit. No-positive WPR is `null`. See [implementation details](docs/IMPLEMENTATION_NOTES.md).

## License and citation

Apache-2.0. `CITATION.cff` identifies the manuscript as a submission preparation, not an accepted conference publication. Author metadata remains anonymous pending an author-approved release.
