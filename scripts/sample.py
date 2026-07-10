"""
sample.py — indlæs et gemt checkpoint og generér tekst.

    python scripts/sample.py --load-dir runs/skjald --prompt "Þat var einn dag"
    python scripts/sample.py --load-dir runs/skjald            # interaktiv løkke

Modellen *fortsætter* din tekst i saga-stil. Dette er tekst-fortsættelse —
ikke spørgsmål/svar. Ægte "svar som en skjald" kræver en finetunet chat-model;
se README-afsnittet "Vejkort: fra saga-stil til samtale".
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Gør stdout robust over for ikke-ASCII (oldnordisk þ/ð/ǫ/æ) på Windows-konsoller.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from nanollm import GPT, GPTConfig, BPETokenizer, generate


def load(load_dir, device):
    ckpt = torch.load(os.path.join(load_dir, "model.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(os.path.join(load_dir, "tokenizer.json"))
    return model, tok


def run(model, tok, text, device, max_new_tokens, temperature, top_k):
    ids = tok.encode(text) or tok.encode(" ")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, max_new_tokens=max_new_tokens,
                   temperature=temperature, top_k=top_k, use_cache=True)
    return tok.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-dir", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.load_dir, device)
    print(f"Indlæste {args.load_dir}  (vocab={tok.vocab_size}, "
          f"kontekst={model.cfg.block_size})\n")

    if args.prompt is not None:
        print(run(model, tok, args.prompt, device,
                  args.max_new_tokens, args.temperature, args.top_k))
        return

    print("Skriv en start-tekst (tom linje eller Ctrl+C for at stoppe):")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            break
        print(run(model, tok, line, device,
                  args.max_new_tokens, args.temperature, args.top_k))
        print()


if __name__ == "__main__":
    main()
