#!/usr/bin/env python3
"""
Build a simple RAG index using scikit-learn.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=str, default="/users/$USER/Legitron/RAG/ihl_index/rules.json")
    parser.add_argument("--model", type=str, default="BAAI/bge-large-en")
    parser.add_argument("--outdir", type=str, default="/users/$USER/Legitron/RAG/ihl_index")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[LOAD] Loading rules...")
    with open(args.rules, "r", encoding="utf-8") as f:
        rules = json.load(f)

    texts = []
    kept_rules = []

    for r in rules:
        rule_text = r.get("rule_text", "").strip()
        if not rule_text:
            continue

        texts.append(rule_text)
        kept_rules.append(r)

    print(f"[INFO] Kept {len(texts)} rules with valid rule_text")

    print(f"[MODEL] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print("[EMBED] Encoding rule texts ONLY...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    np.save(f"{args.outdir}/embeddings.npy", embeddings)

    with open(f"{args.outdir}/rules.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved embeddings to {args.outdir}/")

if __name__ == "__main__":
    main()