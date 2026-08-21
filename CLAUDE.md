# SkjaldTale

Komplet, minimal LLM-stak **fra bunden** i ren PyTorch. Formålet er at *forstå* hver
komponent — ikke at være hurtigst. Ingen fusede kerner, ingen skjult magi.
Alt læsbart og annoteret **på dansk**.

Første led i skjald-slægten: **SkjaldTale** (LLM fra bunden) →
[SkjaldSvar](../SkjaldSvar) (LoRA-finetune) → [Hirdskjald](../Hirdskjald) (lokal 7B + RAG).

## Layout
```
nanollm/        model.py · tokenizer.py · generate.py   # selve pakken
scripts/        prepare_data.py · train.py · sample.py · finetune_lora.py
tests/          test_cache_equivalence.py               # KV-cache == no-cache
data/  runs/
```
`pip install -r requirements.txt` (torch ≥ 2.0). Kør `pytest` for testene.

## Sådan skal her arbejdes
Brugeren har dyb systems-/C#-baggrund men er **ny i ML**. Forklar koncepterne
undervejs — token, attention, KV-cache, quantization, prefill — kør ikke bare
kommandoen. Læringen ER produktet her; en genvej der virker men ikke forklares er
et tab, ikke en gevinst.

`test_cache_equivalence.py` er den vigtigste test i repoet: KV-cachen skal give
**præcis** samme logits som uden cache. Går den i stykker, er generering subtilt
forkert på en måde man ikke ser i outputtet.
