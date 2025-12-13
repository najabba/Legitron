#!/usr/bin/env python3
"""
Query the LLM with RAG - retrieves relevant rules and sends them to the LLM
For use on CSCS
"""

import json
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_rag_index(index_dir: str):
    """Load NumPy embeddings and rules metadata"""
    embeddings_path = os.path.join(index_dir, "embeddings.npy")
    rules_path = os.path.join(index_dir, "rules.json")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}. Run build_rag.py first.")
    
    # Load the raw numpy matrix
    embeddings = np.load(embeddings_path)
    
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    
    print(f"[RAG] Loaded index with {len(embeddings)} vectors")
    return embeddings, rules

def retrieve_context(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    rules: list,
    top_k: int = 5,
) -> str:
    """Retrieve top-k relevant rules for the query"""
    # 1. Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # 2. Compute Cosine Similarity (Query vs All Stored Embeddings)
    # util.cos_sim is optimized and faster than manual numpy math
    corpus_embeddings = torch.from_numpy(embeddings).to(query_embedding.device)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # 3. Get Top-K Scores
    top_results = torch.topk(cos_scores, k=min(top_k, len(rules)))
    
    # 4. Build context
    context_parts = []
    print(f"\n[RAG] Top {top_k} matches:")
    
    for score, idx in zip(top_results.values, top_results.indices):
        idx = int(idx) # Convert tensor to int
        rule = rules[idx]
        print(f"  - [{score:.4f}] Rule {idx}") # Debug print
        context_parts.append(f"[Rule {idx}] {rule['text']}")
    
    context = "\n\n".join(context_parts)
    return context

def build_prompt(query: str, context: str, system_prompt: str = None) -> str:
    """Build the final prompt for the LLM"""
    if system_prompt is None:
        system_prompt = "You are a helpful assistant specialized in international humanitarian law and legal matters. Answer the user's question based on the provided context."
    
    prompt = f"""{system_prompt}

## Context from Knowledge Base:
{context}

## Question:
{query}

There can be multiple correct options.

## Answer:"""
    return prompt


def query_llm(
    prompt: str,
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_length: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Query the LLM with the given prompt"""
    print(f"[LLM] Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("[LLM] Generating response...")
    
    # Apply chat template if available
    messages = [{"role": "user", "content": prompt}]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        # Fallback for models without chat template
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Query LLM with RAG on CSCS")
    parser.add_argument(
        "--index-dir",
        type=str,
        default="/users/$USER/ML_legitron/RAG/ihl_index",
        help="Directory containing embeddings",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en",
        help="Embedding model for retrieval",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        required=True,
        help="Path to fine-tuned LLM model",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of rules to retrieve",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum length of generated response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to save output (default: stdout)",
    )
    
    args = parser.parse_args()
    
    # Expand environment variables in paths
    index_dir = os.path.expandvars(args.index_dir)
    llm_model = os.path.expandvars(args.llm_model)
    
    # Load RAG components
    index, rules = load_rag_index(index_dir)
    embedding_model = SentenceTransformer(args.embedding_model)
    
    # Process question
    question = args.question
    print(f"\n[Q] {question}\n")
    
    # Retrieve context
    context = retrieve_context(question, embedding_model, index, rules, args.top_k)
    print(f"[RAG] Retrieved {args.top_k} relevant rules\n")
    
    # Build prompt
    prompt = build_prompt(question, context)
    
    # Query LLM
    response = query_llm(
        prompt,
        llm_model,
        max_length=args.max_length,
        temperature=args.temperature,
    )
    
    # Output result
    output_text = f"[A] {response}\n"
    print(output_text)
    
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n\n")
            f.write(f"Answer: {response}\n")
        print(f"\n[SAVED] Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
