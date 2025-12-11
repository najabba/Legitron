#!/usr/bin/env python3
"""
Build a simple RAG index using scikit-learn (no FAISS, no Chroma).
Perfect for CSCS HPC environment.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=str, default="rules_clean_text.json")
    parser.add_argument("--model", type=str, default="BAAI/bge-large-en")
    parser.add_argument("--outdir", type=str, default="ihl_index")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[LOAD] Loading rules...")
    with open(args.rules, "r", encoding="utf-8") as f:
        rules = json.load(f)

    texts = [r["text"] for r in rules]

    print(f"[MODEL] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print("[EMBED] Encoding rule texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    np.save(f"{args.outdir}/embeddings.npy", embeddings)

    with open(f"{args.outdir}/rules.json", "w") as f:
        json.dump(rules, f, indent=2)

    print(f"[DONE] Saved index to {args.outdir}/")

if __name__ == "__main__":
    main()