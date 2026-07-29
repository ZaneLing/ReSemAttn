from resemreason.data import load_questions


def test_end_to_end_with_toy_oracle(pipeline):
    examples = load_questions(pipeline._resolve(pipeline.config["data"]["questions"]))
    example = examples[0]
    result = pipeline.infer(
        example.question,
        use_gold_anchors=example.gold_anchors,
        use_gold_schemas=example.gold_schemas,
    )
    candidate_ids = {item.entity_id for item in result.predictions}
    assert "effect:nausea" in candidate_ids
    assert "effect:infection" in candidate_ids
    assert result.predictions
