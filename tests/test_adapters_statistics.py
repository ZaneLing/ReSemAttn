import pytest

from resemreason.adapters import benchmark_outputs, execute_select_chain, parse_select_chain
from resemreason.metrics import average_precision, open_answer_scores, relation_f1
from resemreason.statistics import annotation_agreement, holm_adjust, paired_bootstrap


def test_query_adapter_preserves_inverse_direction(pipeline):
    query = (
        "SELECT ?answer WHERE { ?drug <indication> <disease:ra> . ?drug <side_effect> ?answer . }"
    )
    anchor, chain = parse_select_chain(query)
    assert chain == [("indication", -1), ("side_effect", 1)]
    assert execute_select_chain(pipeline.kg, anchor, chain) == ["effect:infection", "effect:nausea"]
    with pytest.raises(ValueError):
        parse_select_chain("SELECT ?a WHERE { <a> <r> ?a . OPTIONAL {?a <s> ?b} }")
    with pytest.raises(ValueError):
        parse_select_chain("SELECT ?a ?b WHERE { <a> <r> ?a . }")


def test_text_normalization_and_relation_multiset():
    assert open_answer_scores("The chronic myeloid leukemia.", ["chronic myeloid leukemia"]) == {
        "OpenEM": 1,
        "TokenF1": 1,
    }
    assert relation_f1([("r", 1), ("r", 1)], [("r", 1)]) == pytest.approx(2 / 3)
    assert relation_f1([("r", -1)], [("r", 1)]) == 0
    assert average_precision([1, 1], [1, 0]) == 0.5


def test_bootstrap_pairs_questions_and_seeds():
    a = {13: [{"question_id": str(i), "H@1": 1} for i in range(5)]}
    b = {13: [{"question_id": str(i), "H@1": 0} for i in reversed(range(5))]}
    result = paired_bootstrap(a, b, samples=100)
    assert result["difference"] == 1 and result["ci95"] == [1, 1]
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(ValueError):
        paired_bootstrap(a, {42: b[13]}, samples=10)


def test_multiple_choice_and_generator_outputs(pipeline):
    from dataclasses import replace

    from resemreason.data import load_questions
    from resemreason.types import CandidatePrediction, InferenceResult

    e = load_questions(pipeline._resolve(pipeline.config["data"]["questions"]))[0]
    e = replace(
        e,
        options={"A": ("effect:nausea",), "B": ("effect:infection",)},
        gold_option="B",
        gold_texts=("infection risk",),
    )
    result = InferenceResult(
        e.question,
        [],
        [],
        [CandidatePrediction("effect:infection", 3, 0.9, [], [])],
        ["effect:infection"],
    )
    recorded = []

    def generate(q, evidence):
        recorded.append((q, evidence))
        return "infection risk"

    output = benchmark_outputs(e, result, pipeline.kg, generate)
    assert output["MCAccuracy"] == 1 and output["MCMRR"] == 1 and output["TokenF1"] == 1
    assert recorded[0][0] == e.question


def test_source_split_preserves_group_and_duplicate_question(tmp_path):
    import runpy
    from pathlib import Path

    split = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"))[
        "grouped_split"
    ]
    rows = [{"id": str(i), "question": f"q{i}", "source_id": str(i // 2)} for i in range(30)]
    rows[-1]["question"] = rows[0]["question"]
    outputs = split(rows)
    sources = [{r["source_id"] for r in outputs[k]} for k in ["train", "dev", "test"]]
    assert (
        not sources[0] & sources[1] and not sources[0] & sources[2] and not sources[1] & sources[2]
    )
    locations = {r["id"]: s for s, rs in outputs.items() for r in rs}
    assert locations["0"] == locations["29"]


def test_annotation_bootstrap_records_undefined_replicates():
    result = annotation_agreement([1, 1], [1, 1], samples=20)
    assert result["agreement"] == 1 and result["cohen_kappa"] is None
    assert result["undefined_replicates"] == 20
    result = annotation_agreement([1, 1, 0, 0], [1, 0, 1, 0], samples=30)
    assert result["cohen_kappa"] == 0 and result["bootstrap_seed"] == 12345


def test_composition_split_audits_atomic_relation_coverage():
    import runpy
    from pathlib import Path

    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"))[
        "primitive_audit"
    ]
    row = {"gold_schemas": [{"relations": ["r"], "directions": [1]}]}
    assert audit({"train": [row], "dev": [row], "test": [row]})["heldout_relations_known"]
    assert not audit({"train": [], "dev": [row], "test": [row]})["heldout_relations_known"]
