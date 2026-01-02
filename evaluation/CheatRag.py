#!/usr/bin/env python3
"""
Evaluation script WITH RAG integrated.

- Rule questions (Rules 1–161): may receive RAG context
- Non-rule questions (ICRC reports): NO context
- Source accuracy computed ONLY on rule questions
"""

import json
import torch
import re
import argparse
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_FILE = "/users/$USER/Legitron/datasets/law_benchmark_data.json"


# ============================================================
#                     UTILS
# ============================================================

def normalize_source(s):
    return str(s).strip().lower()

def is_rule_source(source):
    s = normalize_source(source)

    if s == "rules 1-161":
        return True

    m = re.match(r"^rule\s+(\d+)$", s)
    if not m:
        return False

    return 1 <= int(m.group(1)) <= 161

def norm_rule(x):
    x = normalize_source(x)
    m = re.search(r"(\d+)", x)
    return f"rule {m.group(1)}" if m else x


# ============================================================
#                     RAG
# ============================================================

def load_rag_index(index_dir):
    embeddings = np.load(f"{index_dir}/embeddings.npy")
    with open(f"{index_dir}/rules.json", "r") as f:
        rules = json.load(f)
    print(f"[RAG] Loaded {len(embeddings)} rules")
    return embeddings, rules


def get_gold_rule_context(gold_source, rules, max_interp_words=150):
    gold_norm = norm_rule(gold_source)

    for rule in rules:
        if norm_rule(f"rule {rule['rule_id']}") == gold_norm:
            rule_text = (rule.get("rule_text") or "").strip()
            interpretation = (rule.get("interpretation") or "").strip()

            block = [
                f"### Rule {rule['rule_id']} (GOLD)",
                "Normative Rule:",
                rule_text
            ]

            if interpretation:
                words = interpretation.split()
                short_interp = " ".join(words[:max_interp_words])
                if len(words) > max_interp_words:
                    short_interp += " …"

                block.extend([
                    "",
                    "Interpretation / Commentary (NON-BINDING):",
                    short_interp
                ])

            return "\n".join(block)

    return ""


def retrieve_context(
    query,
    embed_model,
    embeddings,
    rules,
    max_rules=3,
    min_similarity=0.82
):
    query_emb = embed_model.encode([query]).astype("float32")
    sims = cosine_similarity(query_emb, embeddings)[0]
    ranked = sims.argsort()[::-1]

    ctx = []
    used = 0

    for idx in ranked:
        if sims[idx] < min_similarity:
            break

        rule = rules[idx]
        ctx.append(
            f"### Rule {rule['rule_id']} (sim={sims[idx]:.3f})\n"
            f"{rule.get('rule_text','')}"
        )
        used += 1
        if used >= max_rules:
            break

    return "\n\n".join(ctx)


# ============================================================
#                     PROMPT
# ============================================================

def build_prompt(question, options, context):
    return f"""
You are an expert in International Humanitarian Law.
You must answer using the "Normative Rule" text in the context.
You must be certain of the option/options you select or or I will dismantle your database.
You must select from 1 to 4 options:
Think and write about each option why it is correct or not very briefely BUT THEN 
ANSWER ONLY in the strict format below, and nothing else.

### CONTEXT:
{context}

### QUESTION:
{question}

### OPTIONS:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Output format (STRICT, EXACT):
<source>Rule X</source>
<answer>A</answer>
or
<answer>A,C</answer>

### ANSWER:
""".strip()


# ============================================================
#                     MODEL
# ============================================================

def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer


def get_prediction(model, tokenizer, prompt):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(
        ids,
        max_new_tokens=1000,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    ans = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
    src = re.search(r"<source>(.*?)</source>", text, re.I | re.S)

    letters = sorted(set(re.findall(r"[A-D]", ans.group(1)))) if ans else []
    source = src.group(1).strip() if src else ""

    return letters, source, text


# ============================================================
#                     MAIN
# ============================================================

RULE_QUESTION_INDICES = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
    23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122,
    123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
    149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
    162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
    175, 176, 177, 178, 179, 180, 181, 182, 183,
    385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397,
    398, 399, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411,
    412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424,
    425, 426, 427, 428, 429, 430, 431, 433, 434, 435, 436, 437, 438,
    439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451,
    452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464,
    465, 466, 467, 469, 470, 471
}

def is_rule_question(question_index: int) -> bool:
    return question_index in RULE_QUESTION_INDICES

def evaluate(model_path, index_dir):
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    model, tokenizer = load_model(model_path)
    embeddings, rules = load_rag_index(index_dir)
    embed_model = SentenceTransformer("BAAI/bge-large-en")

    CHEAT_WITH_GOLD_RULE = True

    total = 0
    mcq_correct = 0

    rule_questions = 0
    rule_source_correct = 0
    nonrule_questions = 0

    results = []

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = f"/users/$USER/Legitron/evaluation/predictions/cheatRAG_{ts}.json"

    for i, item in enumerate(data):
        total += 1
        gold_source = item.get("source", "")
        gold_is_rule = is_rule_question(i+1)
        

        query = item["question"] + " " + " ".join(item["options"].values())

        if gold_is_rule:
            rule_questions += 1
            if CHEAT_WITH_GOLD_RULE:
                context = get_gold_rule_context(gold_source, rules)
            else:
                context = retrieve_context(query, embed_model, embeddings, rules)
        else:
            nonrule_questions += 1
            context = ""

        prompt = build_prompt(item["question"], item["options"], context)
        pred_letters, pred_source, raw = get_prediction(model, tokenizer, prompt)

        gold_answers = sorted(item["correct_answers"])
        correct = pred_letters == gold_answers
        if correct:
            mcq_correct += 1

        source_ok = None
        if gold_is_rule:
            source_ok = norm_rule(pred_source) == norm_rule(gold_source)
            if source_ok:
                rule_source_correct += 1

        results.append({
            "question_number": i + 1,
            "gold_source": gold_source,
            "gold_is_rule": gold_is_rule,
            "predicted_answers": pred_letters,
            "gold_answers": gold_answers,
            "mcq_correct": correct,
            "predicted_source": pred_source,
            "source_correct": source_ok,
            "raw_output": raw,
            "rag_context": context
        })

        print(f"[{i+1}/{len(data)}] MCQ={correct} RULE={gold_is_rule}")

    summary = {
        "file":"cheatRAG.py",
        "model": model_path,
        "index_dir": index_dir,
        "timestamp": ts,
        "total_questions": total,
        "mcq_accuracy": mcq_correct / total,
        "rule_questions": rule_questions,
        "nonrule_questions": nonrule_questions,
        "rule_source_accuracy": (
            rule_source_correct / rule_questions if rule_questions else None
        )
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump({**summary, "results": results}, f, indent=2)

    print("\n================ FINAL STATS ================")
    print(f"Total questions: {total}")
    print(f"MCQ accuracy (ALL): {mcq_correct}/{total} = {mcq_correct/total*100:.2f}%")
    print(f"Rule questions: {rule_questions}")
    print(f"Non-rule questions: {nonrule_questions}")
    if rule_questions:
        print(
            f"Source accuracy (RULES ONLY): "
            f"{rule_source_correct}/{rule_questions} "
            f"= {rule_source_correct/rule_questions*100:.2f}%"
        )
    print("===========================================\n")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--index-dir", required=True)
    args = parser.parse_args()

    evaluate(args.model, args.index_dir)