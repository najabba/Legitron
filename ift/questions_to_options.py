import json
import re
import os
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- CONFIGURATION ---
INPUT_FILE = "/capstor/store/cscs/swissai/a127/homes/lsimonnet/axolotl_datasets/charlotteScrape_to_questions_step1.json"
OUTPUT_FILE = "/capstor/store/cscs/swissai/a127/homes/lsimonnet/axolotl_datasets/charlotteScrape_final_dataset_step2.json"

MODEL_PATH = "Qwen/Qwen2.5-32B-Instruct"

TENSOR_PARALLEL_SIZE = 1       
MAX_MODEL_LEN = 16384           

# --- PROMPTS ---
SYSTEM_PROMPT = """You are a Senior International Humanitarian Law (IHL) Expert and Educator.
Your task is to generate high-quality Multiple Choice Questions (MCQ) for a specialized dataset.
You must analyze the provided Source Text and the specific Scenario to create precise legal options and a detailed reasoning trace."""

def final_clean(text: str) -> str:
    """Cleans encoding artifacts."""
    if not text: return ""
    text = text.replace("?~@~Ys", "'s").replace("?~@~Y", "'")
    text = text.replace("?~@~S", "-").replace("?~@~\\", '"')
    text = text.replace("?~@~T", "-")
    return re.sub(r'\?~@~.', '', text)

def prepare_step2_prompts(data: List[Dict], tokenizer) -> List[str]:
    prompts_list = []
    print("⚙️  Formatting prompts...")
    
    for entry in tqdm(data):
        context = final_clean(entry.get("context", ""))
        question = final_clean(entry.get("question", ""))
        
        user_content = f"""
        ### SOURCE TEXT (Legal Context):
        {context} 
        
        ### SCENARIO / QUESTION:
        {question}

        ### INSTRUCTIONS:
        1. Create 4 distinct options (labeled A, B, C, D) for this question.
           - Ensure the options are legally plausible.
           - DESIGN: This is a "Select all that apply" question. Multiple options can be true.
        2. Identify the CORRECT options based on the text and IHL rules.
        3. Write a "Reasoning Trace" (Explanation):
           - Use the IRAC method (Issue, Rule, Application, Conclusion).
           - Base your reasoning primarily on the Source Text.

        ### OUTPUT FORMAT (JSON ONLY):
        {{
            "options": {{
                "A": "Option text A...",
                "B": "Option text B...",
                "C": "Option text C...",
                "D": "Option text D..."
            }},
            "correct_options": ["<List of correct letters, e.g. 'A', 'C' or 'B'>"], 
            "explanation": "Detailed legal reasoning following IRAC structure..."
        }}
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_list.append(full_prompt)
        
    return prompts_list

def extract_json(text: str) -> str:
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return text[start:end]
        return text
    except:
        return ""

def main():
    print(f"📂 Loading Step 1 data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"🔧 Initializing Tokenizer ({MODEL_PATH})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    prompts = prepare_step2_prompts(data, tokenizer)

    print(f"🔌 Initializing vLLM Engine ({MODEL_PATH}) on {TENSOR_PARALLEL_SIZE} GPU...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.98, 
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=2048,
        stop=["<|im_end|>"]
    )

    print("🔥 Generating Options & Answers...")
    outputs = llm.generate(prompts, sampling_params)

    final_dataset = []
    success_count = 0

    print("📥 Parsing outputs...")
    for original_entry, output in zip(data, outputs):
        generated_text = output.outputs[0].text
        json_str = extract_json(generated_text)
        
        try:
            qa_block = json.loads(json_str)
            
            if "options" in qa_block and "correct_options" in qa_block and len(qa_block["correct_options"]) > 0:
                complete_entry = {
                    "source_id": original_entry.get("source_id"),
                    "context": final_clean(original_entry.get("context")),
                    "question": final_clean(original_entry.get("question")),
                    "options": qa_block["options"],
                    "correct_options": qa_block["correct_options"],
                    "explanation": qa_block.get("explanation", ""),
                    "model_used": MODEL_PATH
                }
                final_dataset.append(complete_entry)
                success_count += 1
                print(f"✅ ID {complete_entry['source_id']}: OK")
            else:
                print(f"❌ ID {original_entry.get('source_id')}: Invalid JSON structure")

        except:
            print(f"❌ ID {original_entry.get('source_id')}: JSON Decode Error")
            continue

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"🎉 TEST COMPLETE. Success: {success_count}/{len(data)}")
    print(f"💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()