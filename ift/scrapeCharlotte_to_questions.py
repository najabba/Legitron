import json
import re
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

INPUT_FILE = "/capstor/store/cscs/swissai/a127/meditron/datasets/legitron/charlotte_scrape/output.json"
OUTPUT_FILE = "/capstor/store/cscs/swissai/a127/homes/$USER/axolotl_datasets/charlotteScrape_to_questions_step1.json"

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

TENSOR_PARALLEL_SIZE = 1
MIN_CHAR_LIMIT = 600
MAX_CHAR_LIMIT = 12000 

SYSTEM_PROMPT = """You are a Legal Analyst specialized in International Humanitarian Law (IHL).
Your task is to read legal texts and formulate complex hypothetical scenarios.
These scenarios will later be used to create Multiple Choice Questions (MCQ) with 4 options where MULTIPLE options can be correct.
You do NOT provide the options or the answer. You only formulate the scenario and the question stem."""

def clean_text(text):
    if not text: return ""
    charmap = {'Y': "'", 'X': '"', 'Z': '-', 'T': '-', 'R': '®', 'C': '©'}
    def fix_mojibake(match): return charmap.get(match.group(1), "")
    text = re.sub(r'\?~@~([A-Z])', fix_mojibake, text)
    text = text.replace("\uf02d", "-").replace("n t ", "n't ")
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'Downloaded from.*', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def prepare_prompts(raw_data, tokenizer):
    prompts_list = []  
    metadata_list = [] 
    
    print("⚙️  Filtering data and formatting prompts...")
    
    valid_candidates = []

    for d in tqdm(raw_data, desc="Filtering"):
        raw_text = d.get("text", "")
        clean_content = clean_text(raw_text)
        current_len = len(clean_content)

        if MIN_CHAR_LIMIT < current_len < MAX_CHAR_LIMIT:
            valid_candidates.append({
                "original_entry": d,
                "clean_text": clean_content
            })

    print(f"📊 Stats: Kept {len(valid_candidates)} documents out of {len(raw_data)} based on length constraints ({MIN_CHAR_LIMIT}-{MAX_CHAR_LIMIT} chars).")

    for i, item in enumerate(tqdm(valid_candidates, desc="Tokenizing")):
        context_content = item["clean_text"]
        
        user_content = f"""
        Read the following IHL legal text (Context).
        Formulate a complex hypothetical question (scenario) based strictly on this text.
        
        CONSTRAINTS:
        1. The question must be a scenario involving armed conflict, protected persons, or military objectives.
        2. DESIGN REQUIREMENT: The question must be suitable for a "Select all that apply" Multiple Choice Question format (4 options, multiple correct answers).
           - Example phrasing: "Which of the following statements are true...?", "Identify all the violations in this scenario...", "Which principles were respected...?"
           - Avoid simple Yes/No questions.
        3. Do NOT provide the options or the answer yet. Only provide the question text.
        4. Output strictly a JSON object.

        CONTEXT:
        {context_content}

        OUTPUT FORMAT:
        {{
            "generated_question": "Your complex IHL scenario question here"
        }}
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_list.append(full_prompt)
        
        metadata_list.append({
            "source_id": i, 
            "context": context_content, 
            "full_text": context_content  
        })
        
    return prompts_list, metadata_list

def extract_json(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return text[start:end]
        return text
    except:
        return text

def main():
    print(f"📂 Loading source from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f: raw_data = json.load(f)
    except:
        with open(INPUT_FILE, 'r') as f: raw_data = [json.loads(line) for line in f]
    
    print("🔧 Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    prompts, metadata_list = prepare_prompts(raw_data, tokenizer)
    
    if not prompts:
        print("❌ No documents passed the filter. Exiting.")
        return

    print(f"🚀 Ready to process {len(prompts)} prompts.")

    print(f"🔌 Initializing vLLM Engine ({MODEL_PATH})...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        max_model_len=32000, 
        gpu_memory_utilization=0.90,
        enforce_eager=True
    )


    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        stop=["<|eot_id|>", "<|end_of_text|>"]
    )

    print("🔥 Generating Questions...")
    outputs = llm.generate(prompts, sampling_params)

    intermediate_data = []
    success_count = 0
    
    print("📥 Parsing outputs...")
    for output, meta in zip(outputs, metadata_list):
        generated_text = output.outputs[0].text
        json_str = extract_json(generated_text)
        
        try:
            data = json.loads(json_str)
            question = data.get("generated_question")
            
            if question and len(question) > 20:
                entry = {
                    "source_id": meta["source_id"],
                    "context": meta["context"],      
                    "question": question
                }
                intermediate_data.append(entry)
                success_count += 1
                
        except json.JSONDecodeError:
            continue 

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(intermediate_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 Process Complete. Success rate: {success_count}/{len(prompts)} ({success_count/len(prompts)*100:.1f}%)")
    print(f"💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()