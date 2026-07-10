"""
train.py — hele kæden i ét kørsel.

    1. Træn en BPE-tokenizer på et korpus
    2. Kod korpus til tokens, split i train/val
    3. Træn en nano-GPT på next-token-forudsigelse
    4. Generér tekst fra den trænede model

Kør:  python scripts/train.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanollm import GPT, GPTConfig, BPETokenizer, generate

# --- korpus ---------------------------------------------------------------
# Vigtigt: variation. Et perfekt-periodisk korpus (samme sætning gentaget)
# kollapser BPE til nogle få kæmpe-tokens, og så er der intet at træne på.
# Mange forskellige sætninger holder token-strømmen lang og lærbar.
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

import random as _r
_r.seed(0)
_pool = []
for _ in range(80):
    _shuffled = _SENTENCES[:]
    _r.shuffle(_shuffled)
    _pool.extend(_shuffled)
CORPUS = " ".join(_pool) + " "

VOCAB_SIZE = 512
BLOCK_SIZE = 128
BATCH_SIZE = 32
STEPS = 400
LR = 3e-3
EVAL_EVERY = 100


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, cfg, device, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, cfg.block_size, BATCH_SIZE, device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Enhed: {device}\n")

    # 1. tokenizer
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=VOCAB_SIZE)
    ids = tok.encode(CORPUS)
    ratio = len(CORPUS.encode("utf-8")) / len(ids)
    print(f"Tokenizer: {len(tok.merges)} merges, vocab={tok.vocab_size}")
    print(f"Korpus: {len(CORPUS)} tegn -> {len(ids)} tokens "
          f"(komprimering {ratio:.2f}x)\n")

    # 2. data-split
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # 3. model
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE,
                    n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = GPT(cfg).to(device)
    print(f"Model: {model.num_params():,} parametre "
          f"({cfg.n_layer} lag, {cfg.n_head} hoveder, d={cfg.n_embd})\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # 4. træn
    print("Træner...")
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = get_batch(train_data, cfg.block_size, BATCH_SIZE, device)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            vl = eval_loss(model, val_data, cfg, device)
            print(f"  step {step:4d}  train {loss.item():.3f}  val {vl:.3f}")
    print(f"Færdig på {time.time() - t0:.1f}s\n")

    # 5. generér
    prompt = "Du har magt"
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = generate(model, idx, max_new_tokens=120,
                   temperature=0.8, top_k=40, use_cache=True)
    print("Generering (temp=0.8, top_k=40):")
    print("  " + tok.decode(out[0].tolist()).replace("\n", " "))


if __name__ == "__main__":
    main()
