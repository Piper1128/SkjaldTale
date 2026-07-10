# data/

Trænings-korpora bor her, men **committes ikke** til repo'et (tredjeparts tekst).
Hent dem med `scripts/prepare_data.py`.

## Saga-korpus (Icelandic Saga Database)

```bash
python scripts/prepare_data.py                 # oldnordisk (.on) -> data/corpus.txt
python scripts/prepare_data.py --langs on is   # + moderne islandsk (mere data)
python scripts/prepare_data.py --langs da      # danske oversættelser
```

Kilde: [sagadb.org](https://sagadb.org) — sagaerne er middelalderlige (public domain);
sagadb.org står for digitaliseringen. Sprog-koder: `on` oldnordisk, `is` moderne
islandsk, `da` dansk, `en` engelsk (+ `no`, `de`, `se`, `fr`).
