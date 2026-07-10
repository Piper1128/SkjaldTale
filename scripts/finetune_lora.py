"""
finetune_lora.py — LoRA: finetune en model uden at røre de oprindelige vægte.

Idé: i stedet for at opdatere en stor vægtmatrix W (out x in), fryser vi den
og lærer kun en lille lav-rang tilføjelse dW = B @ A, hvor
    A: (r, in), B: (out, r),  r << min(in, out).
Ved start er B = 0, så dW = 0 og modellen opfører sig præcis som før.
Kun A og B trænes. På rigtige store modeller er det typisk godt under 1%
af parametrene; her (en nano-model med d=128) bliver andelen større, fordi
rang-8-adaptere fylder relativt meget når selve modellen er lille.

Kør:  python scripts/finetune_lora.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from nanollm import GPT, GPTConfig, BPETokenizer


class LoRALinear(nn.Module):
    """Indpakker et frosset nn.Linear og lægger en trænbar lav-rang delta til."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # basis fryses
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))  # start = 0 -> delta = 0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def inject_lora(model: nn.Module, targets=("qkv", "proj", "fc"),
                r: int = 8, alpha: int = 16) -> int:
    """Erstat udvalgte Linear-lag med LoRALinear. Returnér antal erstattede."""
    n = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
                n += 1
    return n


def main():
    torch.manual_seed(0)
    device = "cpu"

    # En lille basis-model (som var den præ-trænet).
    cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(cfg).to(device)

    # Frys HELE basismodellen først — embeddings, positions-embeddings, alle
    # layernorms OG den vægtdelte lm_head. LoRA-adapterne (A/B) oprettes bagefter
    # af inject_lora og er derfor de eneste parametre med requires_grad=True.
    # Uden dette ville optimeren også opdatere embeddings/LN/lm_head (og dermed
    # "røre" basisvægtene, stik imod hele idéen) og oppuste den viste andel af
    # trænbare parametre.
    for p in model.parameters():
        p.requires_grad = False

    # Gem en reference til en basisvægt, så vi kan bevise den ikke ændrer sig.
    ref = model.blocks[0].attn.qkv.weight.detach().clone()

    replaced = inject_lora(model, r=8, alpha=16)
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA injiceret i {replaced} lag (r=8)")
    print(f"Trænbare parametre: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)\n")

    # Lille finetune på tilfældige data — vi tjekker kun at loss falder
    # og at de frosne vægte står stille.
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3)
    data = torch.randint(0, cfg.vocab_size, (4, cfg.block_size + 1))
    print("Finetuner (kun A/B opdateres)...")
    for step in range(60):
        x, y = data[:, :-1], data[:, 1:]
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"  step {step:2d}  loss {loss.item():.3f}")

    now = model.blocks[0].attn.qkv.base.weight
    unchanged = torch.equal(ref, now)
    print(f"\nBasisvægt uændret efter træning: {unchanged} "
          f"(kun A/B blev opdateret)")


if __name__ == "__main__":
    main()
