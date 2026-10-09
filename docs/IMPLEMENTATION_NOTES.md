# Implementation details

Search is discrete, with fixed coefficients. It has no learned extension scorer and no attention-to-search feedback. Grounding scores the full loaded entity inventory in chunks; no persistent embedding cache is used. Paths completed at retained schema hop counts are collected before continuing-beam pruning. Edge attempts include cycle-rejected extensions; unique visited endpoints are a different count.

Path memories mean-pool type and directed-relation embeddings. They do not preserve relation order; positional order is explicitly checked by the mismatch energy. Graph degree is not provenance reliability. K and V have separate low-rank mixtures. Updated token states feed the next of two evidence-attention layers.

Training uses the exact candidate forward used by inference, followed by BCE, positive-set listwise loss and Brier calibration. The appendix's grounding/schema NLL and supplied matched-pair completion losses are enabled. Only discrete retrieval and schema selection are detached. Retained schema probabilities remain differentiable in the evidence prior. No-candidate questions are skipped and counted.

The generator is optional and separate from the entity-ranker objective: generation consumes predicted evidence and the question, never gold answers. Its prompt/decoding completion is documented, not claimed to be a recovered historical implementation.

Scores use log-mean-exp absolute support. Uniformly shifting path compatibilities leaves normalized attention prior unchanged and changes absolute support. Duplicating all paths preserves log-mean-exp; selectively duplicating one path does not. Explanations use argmax-u, not attention sums. In the CE control, explanations instead follow the cross-encoder's path score, as appropriate for that baseline.

Checkpoint format v2 includes model state, grounding temperature, seed, calibrated threshold, effective config, KG vocabulary and exact graph-file hashes. Loading rejects older surrogate-trained checkpoints and mismatched graphs/architectural control variants. Predictions include the checkpoint hash; real evaluation cannot silently use random initialization. Test data cannot fit thresholds through the calibration CLI.

This software implements inspectable evidence scoring. A path's independent biomedical validity must be annotated separately; schema agreement and attention are not causal or clinical validation. No experimental result is generated without actual model outputs.
