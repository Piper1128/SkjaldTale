"""
generate.py
===========
Autoregressiv generering: modellen forudsiger ét token ad gangen, føjer det
til sekvensen og gentager. Her styrer vi *hvordan* det næste token vælges,
og bruger KV-cache så vi ikke genberegner fortiden ved hvert skridt.

Sampling-knapper:
    temperature : >1 = mere kaos, <1 = mere sikker, 0 = grådig (argmax)
    top_k       : behold kun de k mest sandsynlige tokens
    top_p       : behold den mindste gruppe hvis sandsynlighed summer til p (nucleus)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _filter_logits(logits: torch.Tensor, top_k: int | None,
                   top_p: float | None) -> torch.Tensor:
    """Sæt uønskede tokens til -inf inden softmax."""
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Fjern haleenden hvor den kumulative sum passerer p.
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # behold altid top-1
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, use_cache: bool = True) -> torch.Tensor:
    """idx: (B, T) start-tokens. Returnerer (B, T + max_new_tokens)."""
    model.eval()
    block_size = model.cfg.block_size
    # Absolutte positions-embeddings er kun defineret for 0..block_size-1.
    # Med KV-cache kan vi derfor ikke overskride konteksten (ingen glidende
    # vindue uden at genberegne fortidens K/V). Vi klipper ærligt af her.
    if use_cache:
        room = block_size - idx.size(1)
        max_new_tokens = max(0, min(max_new_tokens, room))
    caches = ([{"k": None, "v": None} for _ in range(model.cfg.n_layer)]
              if use_cache else None)

    # Prefill: kør hele prompten igennem én gang for at fylde cachen.
    cur = idx
    pos_offset = 0
    if use_cache:
        cond = idx[:, -block_size:]
        logits, _ = model(cond, caches=caches, pos_offset=0)
        pos_offset = cond.size(1)
        next_tok = _sample(logits, temperature, top_k, top_p)
        cur = torch.cat([idx, next_tok], dim=1)

    for _ in range(max_new_tokens - (1 if use_cache else 0)):
        if use_cache:
            # Decode: kun det seneste token ind, resten ligger i cachen.
            logits, _ = model(cur[:, -1:], caches=caches, pos_offset=pos_offset)
            pos_offset += 1
        else:
            cond = cur[:, -block_size:]
            logits, _ = model(cond, pos_offset=0)
        next_tok = _sample(logits, temperature, top_k, top_p)
        cur = torch.cat([cur, next_tok], dim=1)
    return cur


def _sample(logits: torch.Tensor, temperature: float,
            top_k: int | None, top_p: float | None) -> torch.Tensor:
    logits = logits[:, -1, :]  # sidste position
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    logits = _filter_logits(logits, top_k, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
