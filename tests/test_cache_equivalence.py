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


def test_cache_logits_match_full_forward():
    """Sammenlign LOGITS pr. trin, ikke kun argmax-tokens.

    Token-testen ovenfor er for svag alene: på en tilfældigt initialiseret model
    flytter en ikke-kausal prefill logits ~0.02 (ca. 11 % af deres spredning),
    men ændrer ikke argmax over 40 tokens — så den forbliver grøn med netop den
    bug den skulle fange (ce0da60). Her kræves at cache-stien giver de samme
    logits som en fuld forward over hele sekvensen, ved prefill OG hvert decode.
    """
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=3, n_head=4, n_embd=64)
    model = GPT(cfg)
    model.eval()

    seq = torch.randint(0, cfg.vocab_size, (1, 12))
    caches = [{"k": None, "v": None} for _ in range(cfg.n_layer)]
    with torch.no_grad():
        cached, _ = model(seq, caches=caches, pos_offset=0)  # prefill
        for step in range(20):
            full, _ = model(seq)
            assert torch.allclose(cached, full, atol=1e-5, rtol=0), (
                f"trin {step}: cache-logits afviger fra fuld forward "
                f"(max |diff| = {(cached - full).abs().max().item():.2e}) — "
                "tjek causal-masken i CausalSelfAttention under prefill."
            )
            nxt = full[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            cached, _ = model(nxt, caches=caches, pos_offset=seq.size(1) - 1)


if __name__ == "__main__":
    test_cache_matches_no_cache()
    test_cache_logits_match_full_forward()
    print("OK — cache og no-cache giver identiske tokens og logits (atol=1e-5)")
