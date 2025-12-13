#!/usr/bin/env python3
"""
Query the LLM with RAG
Embeddings are computed ONLY from rule_text.
Interpretations are appended AFTER retrieval.
"""

import json
import argparse
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
#              LOAD RAG INDEX
# ============================================================

def load_rag_index(index_dir: str):
    emb_path = os.path.join(index_dir, "embeddings.npy")
    rules_path = os.path.join(index_dir, "rules.json")

    if not os.path.exists(emb_path):
        raise FileNotFoundError("Embeddings not found. Run build_rag.py first.")

    embeddings = np.load(emb_path)

    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    assert len(embeddings) == len(rules), "Embedding/rule count mismatch"

    print(f"[RAG] Loaded {len(rules)} rules")
    return embeddings, rules


# ============================================================
#              RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    query,
    embed_model,
    embeddings,
    rules,
    top_k=5,
    min_similarity=0.7,
    max_interp_words=200,
):
    query_emb = embed_model.encode([query]).astype("float32")
    similarities = cosine_similarity(query_emb, embeddings)[0]

    # sort indices by similarity (descending)
    ranked_idx = similarities.argsort()[::-1]

    context_parts = []
    used = 0

    for idx in ranked_idx:
        sim = similarities[idx]

        # 🔥 HARD FILTER
        if sim < min_similarity:
            break

        rule = rules[idx]
        rule_id = rule.get("rule_id", "Unknown")
        rule_text = rule.get("rule_text", "").strip()
        interpretation = (rule.get("interpretation") or "").strip()

        block = [
            f"### Rule {rule_id}  (similarity = {sim:.3f})",
            rule_text
        ]

        if interpretation:
            words = interpretation.split()
            short_interp = " ".join(words[:max_interp_words])
            if len(words) > max_interp_words:
                short_interp += " …"

            block.append("\nInterpretation (extract):")
            block.append(short_interp)

        context_parts.append("\n".join(block))
        used += 1

        if used >= top_k:
            break

    return "\n\n".join(context_parts)
# ============================================================
#              BUILD PROMPT
# ============================================================

def build_prompt(query, context):
    system_prompt = (
        "You are an expert in International Humanitarian Law. "
        "Answer the question using ONLY the rules provided below. "
        "Cite rule numbers explicitly in your answer."
    )

    return f"""
{system_prompt}

### CONTEXT:
{context}

### QUESTION:
{query}

### ANSWER:
""".strip()


# ============================================================
#              RUN LLM
# ============================================================

def query_llm(prompt, model_path, max_new_tokens=512, temperature=0.2):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


# ============================================================
#              MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=str, required=True)
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en")
    parser.add_argument("--llm-model", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    embeddings, rules = load_rag_index(args.index_dir)

    embed_model = SentenceTransformer(args.embedding_model)

    context = retrieve_context(
        query=args.question,
        embed_model=embed_model,
        embeddings=embeddings,
        rules=rules,
        top_k=args.top_k,  
)

    print("\n[RAG CONTEXT]\n")
    print(context)

    prompt = build_prompt(args.question, context)

    answer = query_llm(prompt, args.llm_model)

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()