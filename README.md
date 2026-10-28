# SkjaldTale

En komplet, minimal LLM-stak fra bunden i ren PyTorch — bygget for at *forstå*
hver komponent, ikke for at være hurtigst. Ingen genveje, ingen fusede kerner,
ingen skjult magi. Alt er læsbart og annoteret på dansk.

> **Navnet.** En *skjald* er den oldnordiske digter der kvæder og fortæller;
> *tale* er både hvad den gør (taler) og hvad den frembringer (en fortælling).
> En sprogmodel er en moderne skjald — deraf **SkjaldTale**. Pakken indeni
> hedder `nanollm`.

## Struktur

```text
SkjaldTale/
├── nanollm/                        # selve pakken  (import: from nanollm import ...)
│   ├── __init__.py                 # GPT, GPTConfig, BPETokenizer, generate
│   ├── tokenizer.py                # byte-level BPE (train / encode / decode / save / load)
│   ├── model.py                    # decoder-only GPT + GPTConfig, med KV-cache i attention
│   └── generate.py                 # autoregressiv sampling: temperature / top_k / top_p
├── scripts/
│   ├── prepare_data.py             # hent et saga-korpus (sagadb.org) -> data/corpus.txt
│   ├── train.py                    # træn på indbygget demo ELLER --data dit eget korpus
│   ├── sample.py                   # indlæs et checkpoint og generér / tal med modellen
│   └── finetune_lora.py            # LoRA: finetune uden at røre basis-vægtene
├── tests/
│   └── test_cache_equivalence.py   # KV-cache skal matche fuld forward ved temperature=0
└── data/                           # korpora (hentes lokalt, committes ikke)
```

## Kør det

```bash
pip install -r requirements.txt

# Hele kæden: tokenizer trænes, en lille GPT trænes på et korpus, tekst genereres
python scripts/train.py

# LoRA-demo: frys basis-vægtene, træn kun lav-rang-adapterne A og B
python scripts/finetune_lora.py

# Regressionstest: cache-stien skal matche den fulde forward — tokens OG logits pr. trin
python tests/test_cache_equivalence.py        # eller: pytest
```

Kræver kun `torch` (>= 2.0 for `scaled_dot_product_attention`). CPU er nok til nano-skala.

## Træn på dit eget korpus (fx oldnordiske sagaer)

```bash
# 1) hent et saga-korpus (oldnordisk .on) -> data/corpus.txt
python scripts/prepare_data.py

# 2) træn en skaleret model og gem den
python scripts/train.py --data data/corpus.txt --save-dir runs/skjald

# 3) tal med den (tekst-fortsættelse i saga-stil)
python scripts/sample.py --load-dir runs/skjald --prompt "Þat var einn morgin"
```

`prepare_data.py` vælger sprog med `--langs`: `on` (oldnordisk, default), `is`
(moderne islandsk), `da` (dansk), `en` (engelsk). Byte-level-tokenizeren koder
þ/ð/ǫ/æ gratis. `train.py` uden `--data` kører stadig den lille indbyggede demo.

## Vejkort: fra saga-stil til samtale

Vær ærlig om skalaen — der gemmer sig to forskellige mål:

- **Fortælle saga-*stil*** — opnåeligt her. En nano-model lærer sagaernes stil og
  stavemønstre og genererer overbevisende-udseende fragmenter (uden global sammenhæng).
- **Svare på spørgsmål som en skjald** — kræver verdensviden + instruktionsfølgning,
  der først opstår ved *milliarder* af parametre. Vejen dertil er **ikke** at træne
  denne model større, men at **LoRA-finetune en eksisterende lille pretrained model**
  (fx SmolLM2 / Qwen2.5-0.5B) på et saga-korpus + spørgsmål/svar-par — præcis teknikken
  i [`scripts/finetune_lora.py`](scripts/finetune_lora.py), bragt i rigtig skala.

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

## Licens

MIT — se [LICENSE](LICENSE).
