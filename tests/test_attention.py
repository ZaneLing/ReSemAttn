import torch

from resemreason.attention import ReSemAttentionLayer


def test_resem_attention_shapes():
    layer = ReSemAttentionLayer(hidden_dim=32, relation_bases=2, relation_rank=4)
    tokens = torch.randn(7, 32)
    memory = torch.randn(3, 32)
    relations = torch.randn(3, 32)
    prior = torch.tensor([0.6, 0.3, 0.1])
    schema = torch.randn(32)
    candidate = torch.randn(32)
    updated, attention = layer(tokens, memory, relations, prior, schema, candidate, 0.8)
    assert updated.shape == (7, 32)
    assert attention.shape == (7, 3)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(7), atol=1e-5)
