"""Offline API compatibility with the paper's Transformers version; no pretrained weights fetched."""

import pytest
import torch

from resemreason.text_encoder import HuggingFaceTextEncoder


def test_hf_hidden_states_and_generation_without_download():
    pytest.importorskip("transformers")
    from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

    class Tokenizer:
        chat_template = None
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            n = 1 if isinstance(texts, str) else len(texts)
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[1, 3, 4]] * n),
                    "attention_mask": torch.ones((n, 3), dtype=torch.long),
                }
            )

        def decode(self, ids, **kwargs):
            return " ".join(str(i) for i in ids.tolist())

    encoder = HuggingFaceTextEncoder.__new__(HuggingFaceTextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    encoder.tokenizer = Tokenizer()
    encoder.max_length = 32
    encoder.hidden_dim = 16
    batch = encoder.encode(["a", "b"])
    assert batch.token_states.shape == (2, 3, 16) and batch.pooled.shape == (2, 16)
    batch.pooled.square().mean().backward()
    assert encoder.model.model.embed_tokens.weight.grad is not None
    answer = encoder.generate_answer("question", "evidence", max_new_tokens=2)
    assert isinstance(answer, str)
