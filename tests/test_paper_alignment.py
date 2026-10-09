import math
from dataclasses import replace

import pytest
import torch

from resemreason.attention import RelationConditionedProjection, ReSemAttentionLayer
from resemreason.data import load_questions
from resemreason.evaluation import build_row, select_threshold
from resemreason.metrics import (
    answer_row,
    canonical_path_key,
    pair_accuracy,
    precision_at_k,
    summarize_rows,
)
from resemreason.pipeline import ReSemReasonPipeline
from resemreason.trainer import ReSemReasonTrainer


@pytest.fixture
def case(repo_root):
    return ReSemReasonPipeline.from_yaml(repo_root / "configs/case_demo.yaml")


def test_low_rank_projection_matches_matrix_formula():
    torch.manual_seed(5)
    layer = RelationConditionedProjection(8, 3, 2)
    memory, relation = torch.randn(4, 8), torch.randn(4, 8)
    actual = layer(memory, relation)
    omega = layer.mixture(relation).softmax(-1)
    expected = torch.stack(
        [
            (layer.base.weight + sum(omega[i, b] * (layer.u[b] @ layer.v[b].T) for b in range(3)))
            @ memory[i]
            for i in range(4)
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_attention_formula_zero_confidence_and_prior():
    torch.manual_seed(4)
    layer = ReSemAttentionLayer(8, 2, 2, dropout=0).eval()
    h, z, r = torch.randn(4, 8), torch.randn(3, 8), torch.randn(3, 8)
    prior = torch.tensor([0.8, 0.15, 0.05])
    s, a = torch.randn(8), torch.randn(8)
    for confidence in [0.0, 0.7]:
        updated, actual = layer(h, z, r, prior, s, a, confidence)
        expected = (
            layer.query(h) @ layer.key(z, r).T / math.sqrt(8)
            + layer.prior_strength * confidence * (prior + 1e-8).log()
        ).softmax(-1)
        torch.testing.assert_close(actual, expected)
        context = expected @ layer.value(z, r)
        gate = layer.gate(torch.cat([h, context, s.expand_as(h), a.expand_as(h)], -1)).sigmoid()
        torch.testing.assert_close(updated, layer.output_norm(h + gate * context))


def test_real_training_updates_both_attention_layers_and_schema(pipeline):
    trainer = ReSemReasonTrainer(pipeline)
    example = load_questions(pipeline._resolve(pipeline.config["data"]["questions"]))[0]
    loss, _ = trainer.loss_for_example(example)
    loss.backward()
    for name in [
        "layers.0.query.weight",
        "layers.1.query.weight",
        "layers.0.value.u",
        "layers.1.gate.weight",
        "schema_predictor.hop_head.weight",
        "path_prior.gamma",
    ]:
        parameter = dict(pipeline.model.named_parameters())[name]
        assert parameter.grad is not None and parameter.grad.norm() > 0, name
    assert not pipeline.grounder.log_temperature.requires_grad


def test_training_uses_predicted_not_gold_search_schemas(pipeline, monkeypatch):
    example = load_questions(pipeline._resolve(pipeline.config["data"]["questions"]))[0]
    observed = []
    original = pipeline.searcher.search

    def record(question, groundings, schemas, device):
        observed.extend(schemas)
        return original(question, groundings, schemas, device)

    monkeypatch.setattr(pipeline.searcher, "search", record)
    ReSemReasonTrainer(pipeline).loss_for_example(example)
    assert observed and len(observed) == pipeline.config["schema"]["top_m"]
    assert observed != list(example.gold_schemas)
    with pytest.raises(ValueError, match="train"):
        ReSemReasonTrainer(pipeline).loss_for_example(replace(example, split="test"))


def test_same_answer_different_support_and_absolute_score(case):
    example = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    paths = list(example.path_pairs[0])
    m = case.model
    m.eval()
    _, _, topology = m.path_encoder.encode_paths(paths, "cpu")
    scores, prior = m.path_prior(paths, list(example.gold_schemas), topology, m.path_encoder)
    assert scores[0] > scores[1] and (scores <= 0).all()
    duplicate = torch.cat([scores, scores])
    torch.testing.assert_close(
        torch.logsumexp(duplicate, 0) - math.log(4), torch.logsumexp(scores, 0) - math.log(2)
    )
    torch.testing.assert_close((scores - 2).softmax(0), prior)
    assert torch.logsumexp(scores - 2, 0) < torch.logsumexp(scores, 0)
    result = m.forward_candidate(
        example.question, "disease:cml", paths, list(example.gold_schemas), "cpu"
    )
    public = m.score_candidate(
        example.question, "disease:cml", paths, list(example.gold_schemas), "cpu"
    )
    assert public.score == pytest.approx(float(result.score.detach()))
    assert public.supporting_paths[0] == paths[0]
    indices = sorted(range(len(paths)), key=lambda i: -float(result.path_scores[i].detach()))
    for observed, expected in zip(public.attentions, result.attentions):
        torch.testing.assert_close(observed, expected[:, indices].detach())


def test_sigmoid_set_decoder_can_return_zero_or_many(case):
    e = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    case.config["model"]["answer_threshold"] = 0.0
    result = case.infer(
        e.question, use_gold_schemas=e.gold_schemas, use_gold_anchors=e.gold_anchors
    )
    assert len(result.answer_ids) == len(result.predictions)
    for prediction in result.predictions:
        assert prediction.probability == pytest.approx(
            torch.tensor(prediction.score).sigmoid().item()
        )
    case.config["model"]["answer_threshold"] = 1.0
    assert (
        case.infer(
            e.question, use_gold_schemas=e.gold_schemas, use_gold_anchors=e.gold_anchors
        ).answer_ids
        == []
    )


def test_checkpoint_roundtrip_and_seed_threshold(case, tmp_path, repo_root):
    e = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    case.config["seed"] = 13
    case.config["model"]["answer_threshold"] = 0.25
    result = case.infer(e.question)
    checkpoint = tmp_path / "model.pt"
    case.save(checkpoint)
    loaded = ReSemReasonPipeline.from_yaml(repo_root / "configs/case_demo.yaml")
    loaded.load(checkpoint)
    again = loaded.infer(e.question)
    assert loaded.config["seed"] == 13 and loaded.model.answer_threshold == 0.25
    assert [(p.entity_id, p.score) for p in again.predictions] == [
        (p.entity_id, p.score) for p in result.predictions
    ]
    assert again.answer_ids == result.answer_ids


def test_metrics_share_independent_validity_cohort():
    rows = []
    for ranked, gold, validity in [
        (["a"], {"a"}, 1),
        (["a"], {"a"}, 0),
        (["b"], {"a"}, 0),
        (["a"], {"a"}, None),
    ]:
        rows.append({**answer_row(ranked, ranked, gold), "top_path_valid": validity})
    m = summarize_rows(rows)
    assert m["WPR"] == 0.5 and m["Joint@1"] == pytest.approx(1 / 3)
    assert m["Joint@1"] == pytest.approx(m["Cohort.H@1"] * (1 - m["WPR"]))
    assert m["AnnotationCoverage"] == 0.75 and m["Unresolved"] == 1
    assert precision_at_k(["a", "a", "b"], {"a"}, 10) == 0.5
    assert pair_accuracy([(1, 1), (2, 1)]) == 0.5
    assert summarize_rows([{**answer_row([], [], {"a"}), "top_path_valid": 0}])["WPR"] is None


def test_independent_labels_override_schema_proxy(case):
    e = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    r = case.infer(e.question, use_gold_schemas=e.gold_schemas, use_gold_anchors=e.gold_anchors)
    assert build_row(e, r, case.kg, {})["top_path_valid"] is None
    p = r.predictions[0].supporting_paths[0]
    labels = {canonical_path_key(e.question_id, p.nodes, p.relations, p.directions): 0}
    row = build_row(e, r, case.kg, labels)
    assert row["top_path_valid"] == 0 and row["validity_source"] == "independent"


def test_dev_threshold_and_zero_positive_cohort():
    rows = [
        {
            "predictions": [
                {"entity_id": "a", "probability": 0.7},
                {"entity_id": "b", "probability": 0.2},
            ],
            "gold_answers": ["a"],
        }
    ]
    threshold, f1 = select_threshold(rows)
    assert 0.2 < threshold <= 0.7 and f1 == 1
    with pytest.raises(ValueError):
        select_threshold([])


def test_tied_schema_heap_is_deterministic(pipeline):
    for name, parameter in pipeline.schema_predictor.named_parameters():
        if not name.startswith("text_encoder."):
            torch.nn.init.zeros_(parameter)
    pipeline.model.eval()
    a = pipeline.schema_predictor.predict("test", top_m=4)
    b = pipeline.schema_predictor.predict("test", top_m=4)
    assert a == b and sum(s.probability for s in a) == pytest.approx(1)


def test_completed_paths_before_beam_pruning(case):
    e = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    from resemreason.types import GroundingCandidate

    case.searcher.beam_width = 1
    pools = case.searcher.search(
        e.question, [GroundingCandidate("drug:imatinib", 1, 0)], list(e.gold_schemas)
    )
    assert "disease:cml" in pools
    for paths in pools.values():
        for path in paths:
            assert len(path.nodes) == len(set(path.nodes))
    assert case.searcher.last_stats["edge_attempts"] >= case.searcher.last_stats["unique_nodes"]


def test_frozen_pool_does_not_mutate_paths(case):
    from copy import deepcopy

    e = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    g, s, pools = case.retrieve(
        e.question, use_gold_schemas=e.gold_schemas, use_gold_anchors=e.gold_anchors
    )
    before = deepcopy(pools)
    case.infer_pool(e.question, g, s, pools)
    assert pools == before


def test_checkpoint_rejects_protocol_drift(case, tmp_path, repo_root):
    path = tmp_path / "model.pt"
    case.save(path)
    loaded = ReSemReasonPipeline.from_yaml(repo_root / "configs/case_demo.yaml")
    loaded.config["schema"]["top_m"] = 1
    with pytest.raises(ValueError, match="schema protocol"):
        loaded.load(path)


def test_bfs_does_not_run_guided_semantics(case, monkeypatch):
    from resemreason.types import GroundingCandidate

    e = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    case.searcher.strategy = "bfs"

    def forbidden(*args, **kwargs):
        raise AssertionError("BFS must not encode endpoints or score schema compatibility.")

    monkeypatch.setattr(case.text_encoder, "encode", forbidden)
    monkeypatch.setattr(case.searcher, "_compatibility", forbidden)
    pools = case.searcher.search(
        e.question, [GroundingCandidate("drug:imatinib", 1, 0)], list(e.gold_schemas)
    )
    assert pools


def test_false_support_requires_fully_labeled_pool(case):
    e = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    case.config["model"]["answer_threshold"] = 0
    r = case.infer(e.question, use_gold_schemas=e.gold_schemas, use_gold_anchors=e.gold_anchors)
    assert summarize_rows([build_row(e, r, case.kg, {})])["FSR"] is None
    labels = {
        canonical_path_key(e.question_id, path.nodes, path.relations, path.directions): 0
        for prediction in r.predictions
        for path in prediction.supporting_paths
    }
    assert summarize_rows([build_row(e, r, case.kg, labels)])["FSR"] == 1


def test_ce_path_hinge_uses_its_actual_path_score(case, monkeypatch):
    e = load_questions(case._resolve(case.config["data"]["questions"]))[0]
    case.model.reasoner = "cross_encoder"
    trainer = ReSemReasonTrainer(case)
    called = []
    forward = case.model.forward_candidate

    def record(question, candidate_id, paths, *args):
        called.append(paths)
        return forward(question, candidate_id, paths, *args)

    monkeypatch.setattr(case.model, "forward_candidate", record)
    loss, parts = trainer.loss_for_example(e)
    assert parts["path_pairs"] == 1 and list(e.path_pairs[0]) in called
    loss.backward()
    assert case.model.scorer[-1].weight.grad.norm() > 0
