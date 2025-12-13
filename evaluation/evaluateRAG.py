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
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_FILE = "/users/nabbassi/ML_legitron/datasets/law_benchmark_data.json"


# ============================================================
#                     RAG FUNCTIONS
# ============================================================

def load_rag_index(index_dir):
    embeddings = np.load(f"{index_dir}/embeddings.npy")
    with open(f"{index_dir}/rules.json", "r") as f:
        rules = json.load(f)
    print(f"[RAG] Loaded {len(embeddings)} embeddings")
    return embeddings, rules


def retrieve_context(query, embed_model, embeddings, rules, top_k=5):
    query_embed = embed_model.encode(query, convert_to_tensor=True)

    corpus_embeddings = torch.from_numpy(embeddings).to(query_embed.device)
    cos_scores = util.cos_sim(query_embed, corpus_embeddings)[0]

    top_results = torch.topk(cos_scores, k=min(top_k, len(rules)))

    context_parts = []

    for score, idx in zip(top_results.values, top_results.indices):
        idx = int(idx)
        rule = rules[idx]
        context_parts.append(f"[Rule {idx+1}] {rule['text']}")

    return "\n".join(context_parts)

def build_rag_prompt(question, options, context):
    return f"""
You are a legal expert in International Humanitarian Law.
Answer the MCQs based on the provided context.

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
Think quickly step by step and then provide the MCQ answer inside <answer></answer> tags in the format:
<answer>A</answer>
or
<answer>A,C</answer>
or
<answer>A,C,B</answer>
or
<answer>A,C,D,B</answer>
There can be other permutations of the examples provided for the correct answers..

Also provide the IHL source that correspond the best inside <source></source>.
Example: <source>Rule 123</source> (could be multiple sources)
### ANSWER:
"""


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
        max_new_tokens=1024,
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
    predicted_letters = sorted(set(re.findall(r"[A-D]", answer_text.upper())))

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
    out = f"/users/nabbassi/ML_legitron/evaluation/predictions/local_model_results_{timestamp}.json"

    print(f"\nRunning evaluation with RAG on {total} questions...")
    print(f"Saving incremental results to: {out}\n")

    with open(out, "w") as f:
        f.write("[\n")

    for i, item in enumerate(data):
        # ----- RAG retrieval -----
        choices = [f"{key}) {value}" for key, value in item["options"].items()]
        question_prompt = item["question"] + "\n" + "\n".join(choices)
        rag_context = retrieve_context(
            question_prompt, embed_model, embeddings, rules, top_k=5
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

        status_icon = "✅" if correct else "❌"
        print(f"[{i+1}/{total}] Correct={correct} Pred={predicted_letters} True={ground_truth}")

        result = {
            "question": item["question"],
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
