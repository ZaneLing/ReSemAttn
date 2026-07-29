# Implementation notes

## Search is discrete

The top-B beam operation is deliberately outside the gradient path. Training the search component requires supervision over local path extensions, positive paths, and matched reachable-but-invalid negatives. The reference trainer focuses on downstream candidate scoring; add a search-ranking dataloader for full staged training.

## Soft schema supervision

Training data may contain hop count, relation-family, direction, bridge-type, and answer-type labels. These labels are training targets only. Validation and test code should call `SoftSchemaPredictor.predict(question)` without passing benchmark metadata.

## Relation-conditioned projections

`RelationConditionedProjection` implements a base projection plus a mixture of low-rank basis transformations. Mixture weights are generated from the directed relation-sequence representation.

## Path validity

`metrics.schema_path_validity` is useful for unit tests and automatic benchmark checks when a gold relation schema is defined. It is not a replacement for independently annotated biomedical validity in the reachable-but-invalid challenge.

## Scaling to PrimeKG

The toy implementation encodes entity text on demand. For PrimeKG-scale runs:

1. precompute entity text embeddings;
2. store them in a memory-mapped matrix or vector index;
3. cache degree statistics and typed adjacency lists;
4. batch endpoint semantic scoring;
5. batch candidate path memories across candidates;
6. use mixed precision and distributed training for Llama-3.1-8B-Instruct.
