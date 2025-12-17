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

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer

def get_prediction(model, tokenizer, question, options):
    prompt_text = f"""You are a legal expert taking a multiple-choice exam.
Question: {question}

A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Think quickly step by step and Select the best option or options and reply with the final answer inside <answer></answer> tags, here are few examples:
<answer>A</answer>,
<answer>C</answer>,
<answer>B,C,D</answer>,
<answer>A,B,C,D</answer>
"""

    messages = [
        {"role": "user", "content": prompt_text}
    ]
    
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)
    except Exception:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(
        input_ids, 
        max_new_tokens=10, 
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip()
    else:
        answer_text = ""

    predicted_letters = sorted(set(re.findall(r"[A-D]", answer_text.upper())))

    return predicted_letters, response

def evaluate(model_path):
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run the sheet/excel converter first!")
        return

    model, tokenizer = load_model(model_path)

    score = 0
    total = len(data)
    results = []

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"/users/$USER/Legitron/evaluation/predictions/local_model_results_{timestamp}.json"
    print(f"\nStarting evaluation on {total} questions using local checkpoint...\n")

    for i, item in enumerate(data):
        predicted_letters, raw_response = get_prediction(model, tokenizer, item['question'], item['options'])
        ground_truth = sorted(item['correct_answers'])
        
        is_correct = (predicted_letters == ground_truth)
        if is_correct: score += 1
            
        status_icon = "✅" if is_correct else "❌"
        print(f"[{i+1}/{total}] {status_icon} | Pred: {predicted_letters} | True: {ground_truth}")

        results.append({
            "question": item['question'],
            "raw_response": raw_response,
            "predicted": predicted_letters,
            "ground_truth": ground_truth,
    	    "correct": is_correct
        })

    accuracy = (score / total) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
     
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        print(f"Predictions saved at /users/$USER/Legitron/evaluation/predictions/local_model_results_{timestamp}.json")

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
	
    evaluate(args.model)
