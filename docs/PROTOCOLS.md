# Data and experiment protocols

## Canonical KG and questions

Entity rows have `id`, `name`, `type`, optional `aliases`. Edge rows have `source`, `relation`, `target`, optional `direction` (±1), and provenance. Inverse traversal preserves the relation family and negates direction. Degree counts these inverse records.

Question rows have `id`, `question`, `gold_answers`, optional `gold_anchors`, `gold_schemas`, `source_id`, `split` (`train/dev/test`), and `dataset` (`biohopr/primekgqa/medreason`). The natural-language `question` alone enters normal grounding/schema prediction. Gold labels stay in supervision/evaluation.

For each training question, optional `candidate_pairs` contains `[positive_id, negative_id]` pairs selected under a documented matching protocol. Optional `path_pairs` contains:

```json
{
  "negative_type": "same_answer",
  "valid_label": 1,
  "invalid_label": 0,
  "valid": {"nodes": ["e0", "bridge1", "a"], "relations": ["r1", "r2"], "directions": [1, 1]},
  "invalid": {"nodes": ["e0", "bridge2", "a"], "relations": ["r3", "r2"], "directions": [1, 1]}
}
```

Both paths must exist in the loaded KG and share the answer. Label validity independently; a correct endpoint or predicted-schema match does not define a positive path. Match hops, endpoint type, degree and retrieval-score buckets upstream. Absent matched pairs leave the corresponding hinge loss inactive; training logs report the supplied pair count.

## Independent validity JSONL

```json
{"question_id":"q1","nodes":["e0","e1","a"],"relations":["r1","r2"],"directions":[1,1],"valid":1}
```

Use `valid: null` for unresolved cases. Missing predicted paths count as invalid per the paper. Existing but unannotated paths remain unresolved and are excluded from evidence metrics. Global answer H@1 and `Cohort.H@1` are both reported so denominators remain explicit. The historical 0.772/0.112/0.602 triple is not forcibly reconciled or copied into outputs.

## Dataset adapters

- **PrimeKGQA:** `convert_primekgqa.py` reads `id/question/sparql` JSONL and executes supported 1–3-hop single-projection chains against the supplied frozen graph. URI-to-ID mapping is explicit. Branching, aggregates, OPTIONAL/UNION/FILTER, variable predicates, multiple projected variables, and unsupported queries are reported in a manifest. Query gold data are preprocessing/evaluation targets only.
- **Multiple choice:** `options` maps option keys to lists of canonical entity IDs; `gold_option` gives the correct key. An option score is the maximum retrieved candidate score among its mapped entities. No mapped retrieved entity means the option is unranked; no retrieved options yields zero accuracy/MRR.
- **Open answer:** `gold_texts` contains accepted reference strings. `configs/medreason.yaml` uses greedy, evidence-conditioned causal-LM generation. The completion prompt is implemented in `HuggingFaceTextEncoder.generate_answer`; max new tokens defaults to 128. EM/token F1 lowercase, strip ASCII punctuation and English articles, and normalize whitespace. Toy configurations may use canonical entity-name realization; this is recorded by configuration and must not be passed off as a measured generation experiment.

## Leakage and manifests

`prepare_data.py --mode source` groups by source before an 80/10/10 allocation (seed 42). `--mode template` or `--mode composition` groups by explicit template IDs or complete directed relation chains. Questions with multiple chains join connected components; exact duplicate questions and supplied source IDs are kept together across all modes. Group sizes can prevent exact row ratios. The manifest records actual IDs/counts/checksums instead of claiming the paper's historical split.

The manifest audits directed atomic-relation coverage; `--require-known-primitives` rejects unseen atomic relations. Also audit type coverage for composition generalization and source provenance across all splits. These constraints require real source metadata; scripts do not invent it. Training rejects explicit dev/test examples, and dev threshold fitting rejects shared question/source IDs with training.

## Frozen pools and controls

`frozen_pool.py export` stores retrieved anchors, question-only schemas, full path tuples/search scores, graph hashes and the retriever checkpoint. `score` validates question and graph identity and leaves pool bytes untouched. Compare the pool hash, Coverage and CandidateRecall within each retriever block before attributing improvements to the reasoner.

Use `--variant NAME` from `configs/controls.yaml` in both training and evaluation. Train each variant and choose its threshold with the same dev protocol. CE, text-only attention, final-bias, prior-only, KV-only, hard-schema and no-absolute-support controls have explicit code paths. Singleton/all-invalid pools and deliberate schema-error cohorts require independently annotated source paths; do not produce them by relabeling ordinary samples.

For E6, `annotations.py export --rows ... --questions ... --entities ... --edges ... --relation-definitions definitions.json --output ...` removes scores and method identities. The batch includes entity descriptions, candidate, directed path, provided relation definitions and edge provenance. Missing definitions/provenance remain null; supply them before formal annotation. Two annotators fill `valid`; `reconcile` keeps disagreement unresolved until an adjudication is supplied. Agreement includes Cohen’s kappa and 1,000 paired-label bootstrap replicates (seed 12345); duplicate/unknown annotation IDs are rejected. The resulting JSONL can be passed directly to `evaluate.py --validity-labels`.

`analyze_predictions.py --rows ... --strata strata.json` uses a prespecified question-ID→stratum map. Paired comparisons accept JSON entries with `name`, `metric`, `a` and `b`; each method maps seed strings to per-question row files. Equal IDs across every seed/method are required. Evidence comparisons require a common fully annotated cohort. Holm correction covers all comparisons supplied in that invocation.

All-invalid-pool FSR is computed only when every path in that nonempty pool has an independent invalid label. `SingletonFSR` further restricts to a one-path pool. Unlabeled pools do not contribute. Path-pair `negative_type` enables per-type and macro PairAcc; missing type labels leave macro PairAcc undefined.
