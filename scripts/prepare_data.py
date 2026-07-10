"""
prepare_data.py — hent et saga-korpus til træning.

Downloader "all_sagas_text.zip" fra Icelandic Saga Database (sagadb.org),
udpakker det i hukommelsen og samler de valgte sprog til én tekstfil.

Arkivet navngiver filer <saga>.<lang>.txt. Sprog-koder:
    on = oldnordisk (Old Norse-ortografi: þ ð ǫ æ)   <- default, mest autentisk
    is = moderne islandsk (normaliseret stavning)
    da = dansk oversættelse
    en = engelsk oversættelse            (no, de, se, fr findes også)

Eksempler:
    python scripts/prepare_data.py                    # oldnordisk -> data/corpus.txt
    python scripts/prepare_data.py --langs on is      # oldnordisk + moderne islandsk
    python scripts/prepare_data.py --langs da --out data/dansk.txt

Sagaerne er middelalderlige (public domain); sagadb.org står for digitaliseringen.
Korpusset committes IKKE til repo'et — kør dette script for at hente det lokalt.
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

URL = "https://sagadb.org/files/zip/all_sagas_text.zip"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", default=["on"],
                    help="sprog-koder at inkludere (default: on)")
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    print(f"Henter {args.url} ...")
    req = urllib.request.Request(args.url, headers={"User-Agent": "SkjaldTale/1.0"})
    blob = urllib.request.urlopen(req, timeout=180).read()
    print(f"  {len(blob):,} bytes")

    z = zipfile.ZipFile(io.BytesIO(blob))
    langs = set(args.langs)
    chosen, texts = [], []
    for name in sorted(z.namelist()):
        parts = name.rsplit(".", 2)  # <saga>.<lang>.txt
        if len(parts) == 3 and parts[2] == "txt" and parts[1] in langs:
            raw = z.read(name).decode("utf-8", "replace").strip()
            texts.append(raw)
            chosen.append((name, len(raw)))

    if not texts:
        sys.exit(f"Ingen filer matchede sprog {sorted(langs)}. "
                 f"Prøv fx: --langs on is da en")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Tre blanke linjer mellem sagaer = tydelig dokument-grænse for modellen.
    corpus = "\n\n\n".join(texts) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\nValgte {len(chosen)} filer ({sorted(langs)}):")
    for name, n in chosen:
        print(f"  {n:9,d}  {name}")
    print(f"\nSkrev {len(corpus):,} tegn til {args.out}")


if __name__ == "__main__":
    main()
