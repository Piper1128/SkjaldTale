"""
train.py — træn en nano-GPT ende-til-ende.

Uden argumenter: en lille selvstændig demo på et indbygget stoisk korpus
(kører på ~1 min CPU) — uændret fra før.

Med --data: træn på din egen tekstfil (fx et saga-korpus fra prepare_data.py)
og skalér op. Gemte checkpoints kan samples med scripts/sample.py.

    python scripts/prepare_data.py                       # -> data/corpus.txt (oldnordisk)
    python scripts/train.py --data data/corpus.txt --save-dir runs/skjald

    # større/bedre (længere træning — gerne mange steps):
    python scripts/train.py --data data/corpus.txt \
        --vocab-size 4000 --block-size 256 --n-layer 6 --n-embd 256 \
        --steps 5000 --batch-size 32 --save-dir runs/skjald
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Gør stdout robust over for ikke-ASCII (oldnordisk þ/ð/ǫ/æ) på Windows-konsoller.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from nanollm import GPT, GPTConfig, BPETokenizer, generate

# Indbygget mini-korpus (fallback når --data ikke gives): 26 stoiske citater.
_SENTENCES = [
    "Du har magt over dit sind, ikke over ydre begivenheder.",
    "Indse dette, og du vil finde styrke.",
    "Enhver hindring for handling fremmer handlingen.",
    "Det der står i vejen, bliver vejen.",
    "Tænk ikke så meget på hvad du mangler, som på hvad du har.",
    "Lykken i dit liv afhænger af dine tankers kvalitet.",
    "Spild ikke resten af dit liv på tanker om andre.",
    "Se altid tingene for hvad de virkelig er.",
    "Det menneske er frit, som lever som det vil.",
    "Tabet er intet andet end forandring.",
    "Naturen elsker forandring, og intet ondt kommer af den.",
    "Vær som klippen, mod hvilken bølgerne uophørligt bryder.",
    "Om lidt har du glemt alt, og alt har glemt dig.",
    "Handl ikke som om du havde ti tusind år tilbage.",
    "Det bedste svar på uret er ikke at ligne den der begik den.",
    "Lad din vilje være i overensstemmelse med naturen.",
    "Ingen mister noget andet liv end det han lever nu.",
    "Sjælen farves af tankernes farve.",
    "Hvor let er det ikke at vise et roligt sind.",
    "Begynd hver morgen med at sige til dig selv hvad du skal gøre.",
    "Tid er en flod af begivenheder, en voldsom strøm.",
    "Det som ikke gavner bikuben, gavner heller ikke bien.",
    "Vær ligefrem, god og oprigtig i alt hvad du gør.",
    "Den der frygter døden frygter enten intet eller en forandring.",
    "Alt hvad vi hører er en mening, ikke en kendsgerning.",
    "Alt hvad vi ser er et perspektiv, ikke sandheden.",
]


def builtin_corpus():
    import random as _r
    _r.seed(0)
    pool = []
    for _ in range(80):
        s = _SENTENCES[:]
        _r.shuffle(s)
        pool.extend(s)
    return " ".join(pool) + " "


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, block_size, batch_size, device, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(save_dir, model, tok, cfg):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(cfg)},
               os.path.join(save_dir, "model.pt"))
    tok.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"Gemte checkpoint i {save_dir}/ (model.pt + tokenizer.json)")


def parse_args():
    p = argparse.ArgumentParser(description="Træn nano-GPT (SkjaldTale).")
    p.add_argument("--data", help="sti til .txt korpus (ellers indbygget stoisk demo)")
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--block-size", type=int)
    p.add_argument("--n-layer", type=int)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--prompt", default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    saga = args.data is not None
    if saga:
        with open(args.data, encoding="utf-8") as f:
            corpus = f.read()
    else:
        corpus = builtin_corpus()

    # Adaptive defaults: lille til demo, større når man træner på egen data.
    vocab_size = args.vocab_size or (4000 if saga else 512)
    block_size = args.block_size or (256 if saga else 128)
    n_layer = args.n_layer or (6 if saga else 4)
    n_embd = args.n_embd or (256 if saga else 128)
    steps = args.steps or (5000 if saga else 400)
    prompt = args.prompt if args.prompt is not None else (
        "Þat var einn morgin" if saga else "Du har magt")

    print(f"Enhed: {device}")
    print(f"Korpus: {len(corpus):,} tegn "
          f"({'fil ' + args.data if saga else 'indbygget demo'})\n")

    # 1. tokenizer
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size)
    ids = tok.encode(corpus)
    ratio = len(corpus.encode("utf-8")) / len(ids)
    print(f"Tokenizer: {len(tok.merges)} merges, vocab={tok.vocab_size}")
    print(f"  {len(corpus):,} tegn -> {len(ids):,} tokens "
          f"(komprimering {ratio:.2f}x)\n")

    # 2. data-split
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    if len(val_data) <= block_size:
        sys.exit("Korpus for lille til dette block-size — brug mere data "
                 "eller et mindre --block-size.")

    # 3. model
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size,
                    n_layer=n_layer, n_head=args.n_head, n_embd=n_embd,
                    dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"Model: {model.num_params():,} parametre "
          f"({cfg.n_layer} lag, {cfg.n_head} hoveder, d={cfg.n_embd}, "
          f"kontekst={cfg.block_size})\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. træn
    print(f"Træner {steps} steps...")
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, block_size, args.batch_size, device)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == 1:
            vl = eval_loss(model, val_data, block_size, args.batch_size, device)
            print(f"  step {step:5d}  train {loss.item():.3f}  val {vl:.3f}  "
                  f"({time.time() - t0:.0f}s)")
    print(f"Færdig på {time.time() - t0:.1f}s\n")

    # 5. gem
    if args.save_dir:
        save_checkpoint(args.save_dir, model, tok, cfg)

    # 6. generér
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = generate(model, idx, max_new_tokens=200,
                   temperature=0.8, top_k=40, use_cache=True)
    print(f"\nGenerering (prompt={prompt!r}, temp=0.8, top_k=40):")
    print("  " + tok.decode(out[0].tolist()).replace("\n", " "))


if __name__ == "__main__":
    main()
