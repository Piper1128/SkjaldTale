"""
test_cache_equivalence.py
=========================
Regressionstest for KV-cachen.

Ved temperature=0 (grådig argmax) SKAL generering med og uden cache give
præcis de samme tokens: cache-stien er kun en optimering, ikke en anden model.
Divergerer de, er de to stier ikke længere matematisk ens — typisk fordi
causal-masken håndteres forskelligt under prefill (flere tokens ad gangen)
kontra decode (ét token ad gangen). Denne test fanger netop dén fejl.

Kør:  python tests/test_cache_equivalence.py      (eller: pytest tests/)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanollm import GPT, GPTConfig, generate


def test_cache_matches_no_cache():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=3, n_head=4, n_embd=64)
    model = GPT(cfg)
    model.eval()

    # Prompt på flere tokens, så prefill-stien (T>1) faktisk udøves.
    prompt = torch.randint(0, cfg.vocab_size, (1, 12))
    max_new = 40  # holdes inden for block_size (12 + 40 < 64)

    with torch.no_grad():
        cached = generate(model, prompt, max_new_tokens=max_new,
                          temperature=0.0, use_cache=True)
        plain = generate(model, prompt, max_new_tokens=max_new,
                         temperature=0.0, use_cache=False)

    n = min(cached.size(1), plain.size(1))
    assert torch.equal(cached[:, :n], plain[:, :n]), (
        "KV-cache divergerer fra den fulde forward ved temperature=0 — "
        "tjek causal-masken i CausalSelfAttention (prefill skal maskeres kausalt)."
    )


if __name__ == "__main__":
    test_cache_matches_no_cache()
    print("OK — cache og no-cache giver identiske tokens ved temperature=0")
