"""
tokenizer.py
============
Byte-level BPE — præcis samme algoritme som GPT-2/GPT-4, uden optimeringerne.

Basis-ordforrådet er de 256 mulige bytes. Derfra lærer vi hvilke nabopar
der oftest optræder, og smelter dem sammen til nye tokens. Fordi vi starter
på byte-niveau, kan enhver tekst (dansk, emoji, kildekode) altid kodes —
der findes ingen "ukendt" token.

To faser:
    train(text, vocab_size): lær sammensmeltningerne
    encode(text) -> ids:     anvend dem
    decode(ids) -> text:     omvendt, tabsfrit
"""
from __future__ import annotations


def get_stats(ids: list[int], counts: dict | None = None) -> dict:
    """Tæl alle nabopar. [1,2,1,2,3] -> {(1,2):2, (2,1):1, (2,3):1}."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Erstat hver forekomst af 'pair' med det enkelte token 'new_id'.
    [1,2,1,2,3], (1,2), 99 -> [99, 99, 3]."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE. merges: (par) -> nyt id. vocab: id -> bytes."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # -- træning ------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab_size skal rumme mindst de 256 bytes"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for m in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break  # intet tilbage at smelte
            pair = max(stats, key=stats.get)  # hyppigste par
            new_id = 256 + m
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"  merge {m + 1:3d}: {pair} -> {new_id} "
                      f"({self.vocab[new_id]!r}) x{stats[pair]}")

    # -- brug ---------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # Anvend sammensmeltninger i den rækkefølge de blev lært (laveste id først).
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find det par blandt de nuværende, der har lavest merge-index.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # ingen flere par kan smeltes
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
