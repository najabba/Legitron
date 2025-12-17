import json
import re
import os
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- CONFIGURATION ---
INPUT_FILE = "/capstor/store/cscs/swissai/a127/meditron/datasets/legitron/charlotte_scrape/output.json"
OUTPUT_FILE = "/capstor/store/cscs/swissai/a127/homes/lsimonnet/axolotl_datasets/ift_try2_vLLM_charlotte_qwen_full.json"

# Modèle Teacher (Local sur le cluster)
MODEL_PATH = "Qwen/Qwen2.5-32B-Instruct"

# Paramètres Physiques
TENSOR_PARALLEL_SIZE = 1  
CONTEXT_CHAR_LIMIT = 12000

# --- PROMPTS ---
SYSTEM_PROMPT = """You are an expert Legal Reasoning Generator specializing in International Humanitarian Law (IHL).
Your task is to extract complex legal scenarios from provided academic texts and formulate them into reasoning training data.

You must follow the IRAC method (Issue, Rule, Application, Conclusion).
The output must be a valid JSON object."""

def clean_text(text: str) -> str:
    """Nettoyage avancé (Regex) des artefacts d'encodage."""
    if not text: return ""
    
    charmap = {'Y': "'", 'X': '"', 'Z': '-', 'T': '-', 'R': '®', 'C': '©'}
    
    def fix_mojibake(match):
        return charmap.get(match.group(1), "")

    text = re.sub(r'\?~@~([A-Z])', fix_mojibake, text)
    text = text.replace("\uf02d", "-").replace("n t ", "n't ")
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'Downloaded from.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def prepare_prompts(raw_data: List[Dict], tokenizer) -> tuple[List[str], List[int]]:
    """Prépare tous les prompts formatés ChatML en mémoire."""
    prompts_list = []
    original_indices = []

    print("⚙️  Formatting prompts...")
    candidates = [d for d in raw_data if len(d.get("text", "")) > 800]
    
    for i, doc in enumerate(tqdm(candidates)):
        clean_content = clean_text(doc.get("text", ""))
        
        context_snippet = clean_content[:CONTEXT_CHAR_LIMIT]

        user_content = f"""
        Analyze the following legal text excerpt (Context). 
        Create a challenging IHL hypothetical question that can be answered using *only* this text.
        Then, provide a Chain-of-Thought reasoning trace and a final answer.

        CONTEXT:
        {context_snippet}

        OUTPUT FORMAT (JSON):
        {{
            "question": "A specific legal question or scenario based on the text.",
            "reasoning": "Step-by-step reasoning using IRAC format:\\n1. Issue: ...\\n2. Rule: (Cite the text)...\\n3. Application: ...\\n4. Conclusion: ...",
            "answer": "A concise final answer."
        }}
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # Application du template Qwen (<|im_start|>...)
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        prompts_list.append(full_prompt)
        original_indices.append(i)
        
    return prompts_list, original_indices

def main():
    print(f"📂 Loading source from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        with open(INPUT_FILE, 'r') as f:
            raw_data = [json.loads(line) for line in f]
    
    print("🔧 Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Préparation des données
    prompts, indices = prepare_prompts(raw_data, tokenizer)
    print(f"🚀 Ready to process {len(prompts)} documents.")

    # Initialisation du Moteur vLLM
    print(f"🔌 Initializing vLLM Engine (TP={TENSOR_PARALLEL_SIZE})...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=8192, 
        enforce_eager=True # Souvent nécessaire pour Qwen sur certaines versions vLLM
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048, # Suffisant pour IRAC + JSON
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    # Génération Massive (Batch)
    print("🔥 Starting High-Throughput Generation...")
    outputs = llm.generate(prompts, sampling_params)

    # Post-traitement
    final_dataset = []
    success_count = 0
    
    print("📥 Processing outputs...")
    for output, original_idx in zip(outputs, indices):
        generated_text = output.outputs[0].text
        
        try:
            # Nettoyage Markdown (```json ... ```)
            cleaned_json = generated_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_json)
            
            # Formatage propre du raisonnement (String only)
            reasoning_raw = data.get('reasoning', '')
            if isinstance(reasoning_raw, (dict, list)):
                reasoning_str = str(reasoning_raw) # Simplifié, mais le modèle devrait suivre le prompt
            else:
                reasoning_str = str(reasoning_raw)

            entry = {
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are an intelligent legal assistant specialized in International Humanitarian Law. You assume the role of a jurist utilizing the IRAC method."
                    },
                    {
                        "from": "human",
                        "value": data.get('question', 'Error')
                    },
                    {
                        "from": "gpt",
                        "value": f"<think>\n{reasoning_str}\n</think>\n\n{data.get('answer', 'Error')}"
                    }
                ],
                "source_doc_id": original_idx
            }
            final_dataset.append(entry)
            success_count += 1
            
        except json.JSONDecodeError:
            continue # On skip silencieusement les échecs pour ce run massif

    # Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"💾 Saving {len(final_dataset)} valid items to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"🎉 Run Complete. Success rate: {success_count}/{len(prompts)} ({success_count/len(prompts)*100:.1f}%)")

if __name__ == "__main__":
    main()
