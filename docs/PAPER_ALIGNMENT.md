# Paper-to-code alignment

Reference: the current 12-page merged `ReSemReason_main.tex` (main text plus Appendices A–H), revised 2026-10-09. This release implements its method and completion specifications. Historical measurements and descriptions of the old reference package remain historical records.

| Paper component | Implementation / verification |
|---|---|
| §3.2 grounding, top-N, cosine / T=0.1 | `grounding.py`; complete inventory scored in chunks, no persistent embedding cache |
| §3.2 factorized schema components, top-M renormalization, entropy confidence | `schema.py`; stable heap entries and log-space renormalization; separate training NLL |
| §3.3 search; coefficients (1,2,.5,.15) | `search.py`; discrete, recomputed prefix scores; positional direction/type penalties |
| Appendix B.3 completion before beam pruning | `search.py`; repeated-node rejection, global attempt budget, K paths per candidate, no new candidate cap/deduplication |
| §3.4 memory features | `memory.py`; mean-pooled type/relation/direction features plus numerical path/topology summaries |
| §3.4 four mismatches and softplus beta | `memory.py`; normalized completed-path alignment, length/closure, mean log-degree |
| §3.4 low-rank K/V and token/path attention | `attention.py`; separate K/V bases and mixture heads; exact `log(p + epsilon)` bias |
| Figure 2 H0=Encoder(q,a), two residual attention updates | `model.py`; candidate-conditioned token stream, original h_q retained separately |
| §3.5 absolute support and explanation | `model.py`; log-mean-exp u and argmax-u, stable canonical tie breaks |
| §3.6 BCE + .5 list + .1 Brier | `losses.py`; list term skipped when no positives are retrieved |
| Appendix C completion requirements | `trainer.py`; full inference forward, grounding/schema NLL weight 1, supplied matched candidate/path hinges .5/.25, margin .2 |
| Appendix A.3 calibrated multi-answer sets | sigmoid decoding; dev grid .05:.05:.95; no forced top-one fallback; calibrated threshold saved/reloaded |
| Appendix A.3 PrimeKGQA | `adapters.py`, `convert_primekgqa.py`; restricted SELECT-chain parsing/execution, unsupported query manifest, answer sets and directed-relation multiset F1 |
| Appendix A.3 MedReason | canonical option-ID mapping, MC accuracy/MRR, optional evidence-conditioned causal-LM generation and fixed EM/token-F1 normalization |
| §4 / Appendix A.2 evidence metrics | `evaluation.py`, `metrics.py`; one independent validity source/cohort for Joint@1 and WPR; unresolved coverage explicit |
| Appendix D same-answer PairAcc | `score_path_pairs.py`; independently labeled graph-existing paths, same endpoint, strict score comparison, per-negative-type and macro PairAcc |
| E1/E2/E5 | frozen-pool export/rescoring, pool hashes, trainable CE and configurable placement/representation/support interventions |
| E3/E4/E6 | schema diagnostics; grouped split manifests and primitive-coverage audits; blinded annotation, kappa/bootstrap, reconciliation and strata analysis |
| Appendix C.3 statistics | `statistics.py`; matched questions/seeds, seed-averaged differences, question bootstrap seed 12345, 10,000 replicates, centered null test, Holm correction |
| Appendix H accounting | `profile_runtime.py`; 50 warmups / 500 timed queries by default, synchronized CUDA, mean/median/p95, peak memory, edge attempts and unique nodes separately |

## Corrections relative to the original repository

1. The old trainer used `mean(path_memory)` and did not train ReSemAttn. Training now calls the same tensor-valued forward used for inference. Public float conversion occurs after scoring only.
2. Gold schemas are targets for training NLL, not substituted into normal inference or retrieval. Discrete hypothesis selection remains detached; retained hypothesis probabilities are recomputed differentiably during training.
3. Softmax over candidate scores was unsuitable for multi-answer sets. Independent sigmoid probabilities and dev calibration replace it; empty answer sets are allowed.
4. P@K divides by the number actually returned up to K. Duplicate canonical IDs cannot inflate precision/recall.
5. Joint@1 no longer silently uses schema agreement while WPR uses independent labels. Both derive from one contingency; `Joint@1 = Cohort.H@1 × (1 − WPR)` whenever WPR is defined.
6. Attention bias uses the paper's `log(p + epsilon)`, and bf16 backbone states are cast at the fp32 reasoning-head boundary.
7. Equal-score schemas no longer trigger dataclass ordering errors in the enumeration heap; tiny hypothesis probabilities are normalized in log space.
8. Checkpoints preserve seed/threshold/protocol and KG fingerprints. Legacy surrogate-trained checkpoints are rejected with an explicit retraining message.
9. The reported configuration replaces old N=20/M=8/B=64/K=16, rank=32, μ=.5 defaults for the Llama profile.
10. Documentation no longer promises a learned local search scorer or historical staged runs: the paper specifies fixed, discrete search coefficients.

## What cannot be inferred from this release

- No official benchmark files, exact split IDs, human biomedical labels, or original trained weights were found in the supplied project or remote repository.
- Paper table entries, including red simulated controls, are not embedded as computed metrics. They have not been reproduced by this change.
- Full Llama training and generation on the stated four 24-GB GPUs require a validated sharding/offload setup; that historical launcher was not present. This repository supplies a single-process, differentiable method reference, not an untested claim of four-GPU execution.
- The manuscript does not fully specify CE training architecture, free-form generation prompt, all benchmark preprocessing choices, or the exact natural/all-invalid/adversarial cohorts. Executable choices here are documented and require new measurements; they are not backfilled historical details.
- Appendix C.2 and the main text's “released trainer/adapters” limitations refer to the original commit `1b5debf`. Update those manuscript statements when describing this new release; the paper itself is not silently rewritten by a code synchronization.
