# nano-llm

En komplet, minimal LLM-stak fra bunden i ren PyTorch — bygget for at *forstå*
hver komponent, ikke for at være hurtigst. Ingen genveje, ingen fusede kerner,
ingen skjult magi. Alt er læsbart og annoteret på dansk.

## Struktur

Repoet er tænkt med denne træstruktur (på Drev ligger filerne fladt i mappen —
læg dem tilbage i undermapperne når du henter dem ned):

```
nano-llm/
├── nanollm/
│   ├── __init__.py          # from nanollm import GPT, GPTConfig, BPETokenizer, generate
│   ├── tokenizer.py         # byte-level BPE fra bunden (train / encode / decode)
│   ├── model.py             # decoder-only GPT + GPTConfig, med KV-cache i attention
│   └── generate.py          # autoregressiv sampling: temperature / top_k / top_p
└── scripts/
    ├── train.py             # end-to-end: træn BPE -> træn GPT -> generér
    └── finetune_lora.py     # LoRA: finetune uden at røre basis-vægtene
```

## Kør det

```bash
# Hele kæden: tokenizer trænes, en lille GPT trænes på et korpus, tekst genereres
python scripts/train.py

# LoRA-demo: frys basis-vægtene, træn kun lav-rang-adapterne A og B
python scripts/finetune_lora.py
```

Kræver kun `torch`. CPU er nok til nano-skala.

## Datastrømmen (kort)

1. **Tokenizer** (`tokenizer.py`) — tekst → byte-sekvens → BPE-merges → token-id'er.
   Byte-niveau betyder at *alt* kan kodes; der findes ingen "ukendt" token.
2. **Model** (`model.py`) — token- + positions-embedding → N transformer-blokke
   (pre-norm: LayerNorm → kausal self-attention → residual; LayerNorm → MLP →
   residual) → afsluttende LayerNorm → lm_head (vægt-bundet med embedding) → logits.
3. **Attention** — Q·Kᵀ → skalér med √d_head → kausal maske → softmax → vægt V.
   KV-cachen genbruger tidligere K/V under generering.
4. **Generering** (`generate.py`) — logits → temperature/top_k/top_p-filter →
   softmax → sampling → næste token → gentag.
5. **LoRA** (`finetune_lora.py`) — frys W, lær kun dW = B·A med lav rang.
   B starter på 0, så modellen er uændret ved start; kun A og B opdateres.

## Ærlige forbehold

- Dette er et *lærings*-repo. Skalaen (d=128, 4 lag) er nano; en rigtig LLM er
  de samme byggeklodser × mange størrelsesordener i data, compute og bredde.
- BPE'en overfitter på et lille korpus — det er forventet og illustrativt.
- Determinisme: `torch.manual_seed(...)` er sat i scripts for reproducerbare kørsler.
