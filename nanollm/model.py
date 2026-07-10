"""
model.py
========
En decoder-only GPT — samme skabelon som rigtige LLM'er, bare nano-skala.

Datastrøm pr. token:
    id -> token-embedding + positions-embedding
       -> N x [ LayerNorm -> causal self-attention -> +residual
                LayerNorm -> MLP                    -> +residual ]
       -> LayerNorm -> lm_head (vægtdelt med token-embedding) -> logits

"Causal" betyder at token t kun må se token 0..t — aldrig fremtiden.
Det er hele hemmeligheden bag at modellen kan trænes til at forudsige
det næste token på hele sekvensen parallelt.

Attention understøtter en KV-cache: under generering genbruges tidligere
K/V i stedet for at genberegne dem for hvert nyt token.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 128   # maks. kontekstlængde
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        # Ét lineært lag producerer Q, K og V på én gang (billigere).
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: dict | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV-cache: læg de nye K/V bag de gamle. Under "decode" er T==1,
        # så vi genbruger al fortidens K/V frem for at regne dem igen.
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # Causal maske er kun nødvendig når vi behandler >1 token (træning/prefill).
        # Ved ét-token-decode ser den nye position lovligt hele fortiden.
        is_causal = cache is None and T > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer-blok: LN før hvert underlag, residual udenom."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Vægtdeling: token-embedding og output-projektion deler matrice.
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init)
        # Skalér residual-projektioner ned (GPT-2-trick mod eksploderende varians).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # Tæl vægtdeling kun én gang (parameters() de-duplikerer allerede).
        n = sum(p.numel() for p in self.parameters())
        return n - self.pos_emb.weight.numel() * 0  # (delingen er allerede uniq)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                caches: list | None = None, pos_offset: int = 0):
        B, T = idx.shape
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = block(x, cache=cache)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        # Ved generering behøver vi kun sidste position.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None
