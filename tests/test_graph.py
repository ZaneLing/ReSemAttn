def test_inverse_traversal(pipeline):
    forward = pipeline.kg.neighbors("drug:methotrexate")
    assert any(edge.relation == "indication" and edge.target == "disease:ra" for edge in forward)
    inverse = pipeline.kg.neighbors("disease:ra")
    edge = next(edge for edge in inverse if edge.target == "drug:methotrexate")
    assert edge.relation == "indication"
    assert edge.direction == -1
