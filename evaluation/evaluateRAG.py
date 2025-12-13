#!/usr/bin/env python3
"""
Evaluation script WITH RAG integrated.
- Loads index + embeddings once
- For every question: retrieves relevant rules and injects into prompt
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

INPUT_FILE = "/users/vesy/Legitron/datasets/law_benchmark_data.json"


# ============================================================
#                     RAG FUNCTIONS
# ============================================================

def load_rag_index(index_dir):
    embeddings = np.load(f"{index_dir}/embeddings.npy")
    with open(f"{index_dir}/rules.json", "r") as f:
        rules = json.load(f)
    print(f"[RAG] Loaded {len(embeddings)} embeddings")
    return embeddings, rules


def retrieve_context(
    query,
    embed_model,
    embeddings,
    rules,
    max_rules=3,
    min_similarity=0.82,
    max_interp_words=150
):
    query_emb = embed_model.encode([query]).astype("float32")
    similarities = cosine_similarity(query_emb, embeddings)[0]

    ranked_idx = similarities.argsort()[::-1]

    context_parts = []
    used = 0

    for idx in ranked_idx:
        sim = similarities[idx]
        if sim < min_similarity:
            break

        rule = rules[idx]
        rule_id = rule.get("rule_id", "Unknown")

        rule_text = (rule.get("rule_text") or "").strip()
        interpretation = (rule.get("interpretation") or "").strip()

        block = [
            f"### Rule {rule_id} (similarity={sim:.3f})",
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

        context_parts.append("\n".join(block))
        used += 1

        if used >= max_rules:
            break

    return "\n\n".join(context_parts)


def build_rag_prompt(question, options, context):
    return f"""
You are an expert in International Humanitarian Law.

IMPORTANT:
- ONLY the text labeled "Normative Rule" creates legal obligations.
- The "Interpretation / Commentary" EXPLAINS the rule but DOES NOT create new obligations.

LOGICAL CONSTRAINTS (STRICT):
- If the question uses words like "always", "never", "only", or "under all circumstances":
  an option is correct ONLY if it is true in ALL realistic IHL scenarios.
  If it is false in even ONE scenario, it MUST be rejected.

ENUMERATED OBLIGATIONS RULE:
- If a Normative Rule lists multiple duties, ALL listed elements are mandatory.
- Selecting only some of them is INCORRECT.

PROHIBITIONS:
- If a Normative Rule prohibits an act, ANY option allowing that act under ANY condition is FALSE.

SOURCE BINDING:
- The selected answer MUST be justified exclusively by the cited <source>.
- If the cited rule does NOT explicitly support an option, it MUST NOT be selected.

### CONTEXT:
{context}

### QUESTION:
{question}

### OPTIONS:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

### INSTRUCTIONS:
1. Identify the SINGLE rule that directly governs this question.
2. Identify the EXACT sentence in the Normative Rule that answers it.
3. Select ONLY the option or options that match that sentence.
4. Perform the FINAL CHECK before answering.

Output format (STRICT):
<source>Rule X</source>
<answer>Y,Z</answer>

### ANSWER:
""".strip()

# ============================================================
#                   LOAD LOCAL MODEL
# ============================================================

def load_model(path):
    print(f"Loading model from {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer


# ============================================================
#                GENERATE PREDICTION WITH RAG
# ============================================================

def get_prediction(model, tokenizer, question, options, rag_context):
    """Now uses RAG context to build the input prompt."""

    prompt_text = build_rag_prompt(question, options, rag_context)

    messages = [{"role": "user", "content": prompt_text}]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    except:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=300,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    # Extract <answer>
    ans_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    answer_text = ans_match.group(1).strip() if ans_match else ""
    predicted_letters = sorted(set(re.findall(r"[A-D]", answer_text)))

    # Extract <source>
    src_match = re.search(r"<source>(.*?)</source>", response, re.DOTALL | re.IGNORECASE)
    predicted_source = src_match.group(1).strip() if src_match else ""

    return predicted_letters, predicted_source, response


# ============================================================
#                        MAIN EVAL
# ============================================================

def evaluate(model_path, index_dir, embed_model_name="BAAI/bge-large-en"):
    # Load dataset
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load model
    model, tokenizer = load_model(model_path)

    # Load RAG components ONCE
    embeddings, rules = load_rag_index(index_dir)
    embed_model = SentenceTransformer(embed_model_name)

    score = 0
    source_score = 0
    total = len(data)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = f"/users/vesy/Legitron/evaluation/predictions/local_model_results_{timestamp}.json"

    print(f"\nRunning evaluation with RAG on {total} questions...")
    print(f"Saving incremental results to: {out}\n")

    with open(out, "w") as f:
        f.write("[\n")

    for i, item in enumerate(data):
        # ----- RAG retrieval -----
        full_query = item["question"] + " " + " ".join(item["options"].values())

        rag_context = retrieve_context(
            full_query, embed_model, embeddings, rules
        )

        predicted_letters, predicted_source, raw_output = get_prediction(
            model, tokenizer, item["question"], item["options"], rag_context
        )

        ground_truth = sorted(item["correct_answers"])
        correct = (predicted_letters == ground_truth)

        if correct:
            score += 1

        source_correct = (
            predicted_source.strip().lower() ==
            item.get("source", "").strip().lower()
        )

        if source_correct:
            source_score += 1

        print(f"[{i+1}/{total}] Correct={correct} Pred={predicted_letters} True={ground_truth}")

        result = {
        "question_number": i + 1,
        "question": item["question"],
        "options": {
            "A": item["options"]["A"],
            "B": item["options"]["B"],
            "C": item["options"]["C"],
            "D": item["options"]["D"]
        },
        "predicted": predicted_letters,
        "ground_truth": ground_truth,
        "source_pred": predicted_source,
        "source_true": item.get("source", ""),
        "correct": correct,
        "source_correct": source_correct,
        "raw_output": raw_output,
        "rag_context_used": rag_context,
}

        with open(out, "a") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            if i < total - 1:
                f.write(",\n")
            else:
                f.write("\n")

    with open(out, "a") as f:
        f.write("]")

    print("\nFINAL RESULTS:")
    print(f"MCQ accuracy: {score}/{total} = {score/total*100:.2f}%")
    print(f"Source accuracy: {source_score}/{total} = {source_score/total*100:.2f}%")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--index-dir", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.model, args.index_dir)