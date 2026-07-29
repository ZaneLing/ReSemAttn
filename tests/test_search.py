from resemreason.types import GroundingCandidate, RelationSchema


def test_relation_guided_search_finds_effects(pipeline):
    schemas = [
        RelationSchema(
            hops=2,
            entity_types=("disease", "drug", "effect"),
            relations=("indication", "side_effect"),
            directions=(-1, 1),
            probability=1.0,
        )
    ]
    groundings = [GroundingCandidate("disease:ra", 1.0, 0.0)]
    found = pipeline.searcher.search(
        "Which adverse effects are linked to drugs used to treat rheumatoid arthritis?",
        groundings,
        schemas,
        pipeline.device,
    )
    assert "effect:nausea" in found
    assert "effect:infection" in found
