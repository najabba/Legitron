import json
import torch
import re
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any

INPUT_FILE = "/users/lsimonnet/meditron/axolotl_config/Legitron/datasets/law_benchmark_data.json"



class ThoughtGraph:
    """
    A minimal in-memory representation of a Graph of Thoughts.
    Each node is a partial reasoning step, edges show how thoughts relate, combine, or refine each other.
    """
    def __init__(self):
        self.nodes = {}          # node_id -> {"text": str, "score": float}
        self.edges = []          # list of (source_id, target_id, relation_type)
        self.next_id = 0

    def add_node(self, text: str, score: float) -> int:
        node_id = self.next_id
        self.nodes[node_id] = {"text": text, "score": score}
        self.next_id += 1
        return node_id

    def add_edge(self, src: int, dst: int, relation: str):
        self.edges.append((src, dst, relation))

    def top_nodes(self, k=3):
        return sorted(
            self.nodes.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:k]


def llm_generate(prompt: str, model, tokenizer, max_new_tokens=96, temperature=0.7) -> str:
    """Simple helper to generate text."""

    messages = [{"role": "user", "content" : prompt}]
    try: input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    except:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response


def generate_next_thought(question: str, model, tokenizer, options, context: str) -> str:
    """ Generate a single next thought that may refine or extend previous thoughts."""
    prompt = (f"""You are an expert in International Humanitarian Law (IHL) and a legal reasoning assistant.
    
Question: {question}

Options: 
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Current reasoning graph context: 
{context}

Generate ONE specific legal reasoning step that: 
- Identifies relevant IHL rules (Geneva Conventions, Additional Protocols, Customary Law)
- Applies these rules to the scenario
- Considers one aspect of the legal analysis

Reasoning step:"""
    )
    result = llm_generate(prompt,model,tokenizer, max_new_tokens=150)
    return result.split("Reasoning step:")[-1].strip()


def score_thought(question: str, options, thought: str, model, tokenizer) -> float:
    """Score how useful a thought is using a simple value prompt."""
    prompt = (f""" Evaluate this IHL legal reasoning step for accuracy and relevance

    Question: {question}

    Options: 
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Reasoning step: {thought}

Rate from 1-10 based on:
- Legal accuracy (correct IHL rules)
- Logical coherence  
- Relevance to question
- Citation quality

Score (number only):"""
    )
    result = llm_generate(prompt, model, tokenizer, max_new_tokens=8, temperature=1e-8)
    raw = result.split("Score:")[-1].strip().split()[0]
    try:
        return float(raw)
    except ValueError:
        return 5.0


def graph_of_thoughts_reason(
    model,
    tokenizer,
    question: str,
    options,
    iterations: int = 5,
    expansions_per_step: int = 3
) -> Dict[str, Any]:
    """
    Build a Graph of Thoughts:
    - Create nodes for new thoughts
    - Score them
    - Link them through refinement or aggregation relationships
    - Use the graph's highest scoring nodes to guide further reasoning
    """
    #Initalize the graph
    graph = ThoughtGraph()

    # Start with an empty initial node
    root_id = graph.add_node("Initial reasoning begins.", score=5.0)

    for i in range(iterations):
        # Create a combined context from top graph nodes
        top_context = "\n".join(
            f"- {n['text']} (score {n['score']})"
            for _, n in graph.top_nodes(k=3)
        )

        for _ in range(expansions_per_step):
            new_thought = generate_next_thought(question, model, tokenizer, options, top_context)
            new_score = score_thought(question, options, new_thought, model, tokenizer)
            new_id = graph.add_node(new_thought, new_score)

            # Link new node to top scoring node (refinement or aggregation)
            best_node_id, _ = graph.top_nodes(k=1)[0]
            relation = "refines" if new_score >= graph.nodes[best_node_id]["score"] else "extends"
            graph.add_edge(best_node_id, new_id, relation)

    # Build final answer from top nodes
    final_outline = "\n".join(
        f"- {n['text']}"
        for _, n in graph.top_nodes(k=5)
    )

    final_prompt = f"""You are an expert in International Humanitarian Law (IHL) taking an exam.

Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Based on this comprehensive legal analysis:
{final_outline}

Provide your final answer using IRAC structure (Issue-Rule-Application-Conclusion).

Format:
<analysis>
Issue: [Legal question]
Rule: [Applicable IHL provisions] 
Application: [Apply rules to facts]
Conclusion: [Final determination]
</analysis>

<answer>A</answer> or <answer>A,C</answer>"""

    final_answer = llm_generate(final_prompt, model, tokenizer, max_new_tokens=512, temperature=0.2)

    return {
        "graph_nodes": graph.nodes,
        "graph_edges": graph.edges,
        "final_answer": final_answer
    }

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

def get_got_prediction(raw_output:str):
    match = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL | re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip()
    else:
        answer_text = ""  # No answer found

    predicted_letters = sorted(set(re.findall(r"[A-D]", answer_text.upper())))

    return predicted_letters


def evaluate(model_path, iterations, expansions_per_step):
    # Load dataset
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load model
    model, tokenizer = load_model(model_path)

    score = 0
    total = len(data)
    results = []

    print(f"\nStarting evaluation on {total} questions...\n")

    for i, item in enumerate(data):
        """
        predicted_letters, raw_output = get_prediction(
            model, tokenizer, item["question"], item["options"]
        )

        """
        got_reason = graph_of_thoughts_reason(model, tokenizer, item["question"], item["options"], iterations, expansions_per_step)


        graph_nodes = got_reason["graph_nodes"]
        graph_edges = got_reason["graph_edges"]
        raw_output = got_reason["final_answer"]

        predicted_letters = get_got_prediction(raw_output)



        ground_truth = sorted(item["correct_answers"])

        is_correct = (predicted_letters == ground_truth)
        if is_correct:
            score += 1

        icon = "✅" if is_correct else "❌"
        print(f"[{i+1}/{total}] {icon} Pred: {predicted_letters} | True: {ground_truth}")

        results.append({
            "question": item["question"],
            "raw_response": raw_output,
            "predicted": predicted_letters,
            "ground_truth": ground_truth,
            "correct": is_correct
        })

    accuracy = (score / total) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = f"/users/lsimonnet/meditron/axolotl_config/Legitron/evaluation/predictions/local_model_results_{timestamp}.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {out}")

    return accuracy



if __name__ == "__main__":
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations on the GoT inference")
    parser.add_argument("--expansions", type=int, required=True, help="Number of exapnsions per step for the GoT")
    args = parser.parse_args()
    evaluate(model_path = args.model, iterations = args.iterations, expansions_per_step = args.expansions)
