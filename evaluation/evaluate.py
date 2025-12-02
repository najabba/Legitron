import json
import torch
import re
import argparse
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_FILE = "/users/$USER/Legitron/datasets/law_benchmark_data.json"

def load_model(path):
    print(f"Loading model from {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Load model with automatic device mapping (GPU if available)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer

def get_prediction(model, tokenizer, question, options):
    # 1. Format the prompt clearly
    # We use a chat template format which works best for Instruct/Chat models
    prompt_text = f"""You are a legal expert taking a multiple-choice exam.
Question: {question}

A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Return ONLY the letter(s) of the correct answer (e.g. 'A' or 'A, C')."""

    messages = [
        {"role": "user", "content": prompt_text}
    ]
    
    # Apply the model's specific chat template (handles special tokens automatically)
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)
    except Exception:
        # Fallback if model has no chat template (base models)
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

    # 2. Generate Response
    # We limit max_new_tokens to 10 because we only want a short answer (A, B, C...)
    outputs = model.generate(
        input_ids, 
        max_new_tokens=10, 
        temperature=0.1, # Low temperature for deterministic answers
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # 3. Decode output
    # Slice [input_ids.shape[1]:] to remove the prompt from the output
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def evaluate(model_path):
    # Load Data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run the sheet/excel converter first!")
        return

    # Load Model
    model, tokenizer = load_model(model_path)

    score = 0
    total = len(data)
    results = []

    print(f"\nStarting evaluation on {total} questions using local checkpoint...\n")

    for i, item in enumerate(data):
        raw_response = get_prediction(model, tokenizer, item['question'], item['options'])
        
        # Extract Letters (A, B, C, D) from response
        # Using Regex to ignore extra text like "The answer is A"
        predicted_letters = sorted(list(set(re.findall(r'[A-D]', raw_response.upper()))))
        ground_truth = sorted(item['correct_answers'])
        
        # Check correctness
        is_correct = (predicted_letters == ground_truth)
        if is_correct: score += 1
        #sample_score = 0
        #for letter in ['A','B','C','D']:
        #    in_prediction = letter in predicted_letters
        #    in_ground_truth = letter in ground_truth
        #    
        #    # Match: Both have it (True Positive) OR Both don't have it (True Negative)
        #    if in_prediction == in_ground_truth:
        #        sample_score += 0.25

        #score += sample_score
	
        
        # Print live status
        status_icon = "✅" if is_correct else "❌"
        #status_icon = "✅" if sample_score == 1 else "❌" if sample_score == 0 else "🟡"
        print(f"[{i+1}/{total}] {status_icon} | Pred: {predicted_letters} | True: {ground_truth}")

        results.append({
            "question": item['question'],
            "raw_response": raw_response,
            "predicted": predicted_letters,
            "ground_truth": ground_truth,
            #"score": sample_score,
	    "correct": is_correct
        })

    accuracy = (score / total) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
        
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"/users/$USER/Legitron/evaluation/predictions/local_model_results_{timestamp}.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    return accuracy

if __name__ == "__main__":
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
	
    evaluate(args.model)
