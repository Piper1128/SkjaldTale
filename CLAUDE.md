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

## Betalte fælder (fund 2026-10-28)
Begge fundet ved at genindføre den fejl kontrollen vogter og se om den blev rød.
Den metode er reglen her: en kontrol der ikke KAN fejle læses som en der består.

- **Cache-testen sammenlignede kun argmax-tokens.** Med ce0da60-buggen genindført
  (`is_causal = T > 1 and cache is None`) flyttede logits ~0.018, men argmax på en
  random-init model ændrede sig ikke over 40 tokens → testen var grøn. Nu kræver
  `test_cache_logits_match_full_forward` `allclose(atol=1e-5)` pr. trin og bliver
  rød på den mutation (cb58741). Fjern ikke logit-sammenligningen.
- **LoRA-demoens "basisvægt uændret" tjekkede én vægt** (`blocks[0].attn.qkv`), som
  `LoRALinear` fryser selv. Uden den globale frys (aee4edb) trænede embeddings/LN/
  lm_head med (15.18 % trænbare), og tjekket sagde stadig True. Nu tjekkes at alle
  trænbare parametre er A/B og at alle 52 basisparametre er uændrede; scriptet
  afslutter med exit 1 ellers (8923c6d).

