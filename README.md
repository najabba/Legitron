# Legitron: Exploratory Framework for International Humanitarian Law (IHL) LLMs

This repository contains the foundational research contributing to **Legitron**, an initiative at the EPFL Light Lab to develop a domain-specialized Large Language Model for International Humanitarian Law.

While the ultimate goal of Legitron is a fully deployable legal assistant, this project focuses on the **comparative analysis and validation of training methodologies**. We explore various base models, fine-tuning strategies (Instruction Fine-Tuning), and Retrieval-Augmented Generation (RAG) pipelines on a targeted scale. The insights derived here serve as the empirical groundwork for the future full-scale development of Legitron.

## 📂 Project Structure

* **`analysis/`**: **Scripts used for analysis purpose**.
    * `Analyze_questions.ipynb`: It reproduces figures and tables found in the report and includes additional exploratory analysis on dataset themes and source distributions that were used to profile the benchmark.
* **`report/`**: **Project Documentation**.
    * `Report.pdf`: The final report detailing our methodology, experiments, and findings.
* **`datasets/`**: **Data Central**. Contains benchmarks, IHL rules, and synthetic training data.
    * `law_benchmark_data.json`: The Golden IHL Benchmark for evaluation (MCQs).
    * `rules_with_interpretations.json`: The corpus of IHL rules used to build the RAG embeddings.
    * `charlotteScrape_*.json`: Intermediate files from the synthetic scenario generation pipeline.
    * `ift_vLLM_charlotte_*.json`: Final Instruction Fine-Tuning (IFT) datasets containing Chain-of-Thought (CoT) reasoning.
* **`ift/`**: **Instruction Fine-Tuning Data Generation Pipeline**.
    * `scrapeCharlotte_to_questions.py`: Step 1 - Converts raw legal texts into complex scenarios.
    * `questions_to_options.py`: Step 2 - Generates options, answers, and IRAC reasoning for the scenarios.
    * `irac_only_ift.py`: Generates Chain-of-Thought (CoT) training data in ChatML format.
    * `ift_implementation.sh`: Slurm launch script for generation jobs.
* **`RAG/`**: **Retrieval-Augmented Generation**.
    * `build_rag.py`: Creates embeddings from the IHL rules.
    * `query_rag.py`: Interactively query the model with RAG support.
    * `ihl_index/`: Stores the generated vector store (embeddings + rules).
* **`evaluation/`**: **Benchmarking Tools**.
    * `CheatRag.py`: .
    * `evaluate.py`: Standard MCQ evaluation on the IHL benchmark.
    * `evaluateRAG.py`: Evaluation using the RAG retriever to inject context.
* **`train/`**: **Training Scripts**.
    * `train.sh`: Launch script for Axolotl training on clusters.
* **`config/`**: **Configuration**.
    * `axolotl_model_datasets_template.yaml`: The master template for all training runs. **Start here.**

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/najabba/Legitron.git
    cd Legitron
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `ift/` scripts require `vllm`. Ensure your environment supports it (CUDA 12+ recommended).*

## 🔧 Environment Setup (CSCS Cluster)

To run experiments on the CSCS cluster, you must configure the Enroot container environments by creating specific TOML configuration files in your local `.edf/` directory.

### 1. Standard Training Environment (`axolotl.toml`)
For standard Continual Pre-Training (CPT) and Fine-Tuning tasks, create a file at `.edf/axolotl.toml` with the following configuration. This uses the standard `axolotl-apertus` image.

```toml
# File: .edf/axolotl.toml

# Base Docker image for training
image = "/capstor/store/cscs/swissai/a127/meditron/docker/axolotl-apertus.sqsh"

mounts = ["/capstor", "/iopsstor", "/users"]
writable = true

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
HF_HOME = "${SCRATCH}/hf"
CUDA_CACHE_DISABLE = "1"
NCCL_NET = "AWS Libfabric"
NCCL_CROSS_NIC = "1"
NCCL_NET_GDR_LEVEL = "PHB"
FI_CXI_DISABLE_HOST_REGISTER = "1"
FI_MR_CACHE_MONITOR = "userfaultfd"
FI_CXI_DEFAULT_CQ_SIZE = "131072"
FI_CXI_DEFAULT_TX_SIZE = "32768"
FI_CXI_RX_MATCH_MODE = "software"
FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD = "16777216"
FI_CXI_COMPAT = "0"
```
### 2. IFT & Inference Environment (`axolotl_vllm.toml`)

For the Instruction Fine-Tuning (IFT) pipeline and synthetic data generation, we require vLLM support. Create a copy named `.edf/axolotl_vllm.toml` and update the `image` path to the vLLM-compatible container:

```toml
# File: .edf/axolotl_vllm.toml

# CHANGED: Uses the vLLM-compatible image
image = "/capstor/store/cscs/swissai/a127/meditron/docker/axolotl-vllm.sqsh"

# ... (Keep all other mounts, annotations, and env variables exactly the same as above)
mounts = ["/capstor", "/iopsstor", "/users"]
writable = true

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
HF_HOME = "${SCRATCH}/hf"
CUDA_CACHE_DISABLE = "1"
NCCL_NET = "AWS Libfabric"
NCCL_CROSS_NIC = "1"
NCCL_NET_GDR_LEVEL = "PHB"
FI_CXI_DISABLE_HOST_REGISTER = "1"
FI_MR_CACHE_MONITOR = "userfaultfd"
FI_CXI_DEFAULT_CQ_SIZE = "131072"
FI_CXI_DEFAULT_TX_SIZE = "32768"
FI_CXI_RX_MATCH_MODE = "software"
FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD = "16777216"
FI_CXI_COMPAT = "0"
```
### Usage

As described in the [LiGHT Lab CSCS Guide](https://epflight.github.io/LiGHT-doc/clusters/cscs/axolotl_training/). setup guide, ensure that your submission scripts or repository configuration files point to the correct TOML file depending on the task:

Use `.edf/axolotl.toml` for standard training.

Use `.edf/axolotl_vllm.toml` for high-throughput generation and IFT scripts.

## 📊 Data Dictionary

| File Name | Description | usage |
| :--- | :--- | :--- |
| **`law_benchmark_data.json`** | Manually curated IHL Multiple Choice Questions. | **Evaluation** (Ground Truth) |
| **`rules_with_interpretations.json`** | Full text of IHL rules with commentaries. | **RAG** (Knowledge Base) |
| **`ift_vLLM_charlotte_qwen_full.json`** | Synthetic training samples with `<think>` tags (IRAC reasoning). | **Training** (Instruction Tuning) |
| **`charlotteScrape_to_questions_step1.json`** | Raw scenarios extracted from legal texts (Output of Step 1). | **Data Gen** (Intermediate) |
| **`charlotteScrape_final_dataset_step2.json`** | Scenarios paired with MCQs and explanations (Output of Step 2). | **Data Gen** (Finalizing) |

## 🚀 Usage Guide

### 1. Data Setup & Training Configuration

To train a model, you must create a configuration file derived from our template.

**Step A: Create your config**
Copy the template to a new file:
```bash
cp config/axolotl_model_datasets_template.yaml config/my_training_run.yaml
```

**Step B: Complete the configuration**
Open `config/my_training_run.yaml` and fill in the missing fields (`base_model`, `datasets`, `output_dir`).

* **To Reproduce Report Results:**
    Most datasets used to obtain the results in our report are hosted directly on the CSCS cluster. **You do not need to regenerate them to reproduce our training runs.**
    
    * **CSCS Access:** To access the server, follow the [LiGHT Lab CSCS Guide](https://epflight.github.io/LiGHT-doc/clusters/cscs/cscs/).
    * **Dataset Path:** The pre-processed datasets (including the "Charlotte" scrape and instruction mixtures) are located in the shared directory:
        ```
        /capstor/store/cscs/swissai/a127/meditron/datasets/legitron/
        ```
    * **Action:** Ensure your Axolotl config files (in `config/`) point to these absolute paths.

* **To Use New Synthetic Data:**
  For future improvements, we have developed a pipeline to generate *new* synthetic instruction data from raw legal texts. This is intended for expanding the training corpus beyond the scope of the original report.
    If you have generated new data using the `ift/` pipeline (see below in 2.), point the `path` to your local JSONL files.

**Step C: Launch Training**
Submit the job to Slurm using your new config:
```bash
sbatch train/train.sh --config config/my_training_run.yaml
```

### 2. Generating Synthetic Data (Future Development) ###

For future improvements, we have developed a pipeline to generate new synthetic instruction data from raw legal texts.

**Option 1: Generate MCQs (Two-Step Process)**

* **Generate Scenarios:** Extracts complex scenarios from raw text.
    ```bash
    python ift/scrapeCharlotte_to_questions.py
    ```
* **Generate Options & Reasoning:** Creates full MCQs with IRAC explanations.
    ```bash
    python ift/questions_to_options.py
    ```

**Option 2: Generate IRAC Chain-of-Thought Data**

Creates training samples with `<think>` tags containing the IRAC reasoning process.
```bash
python ift/irac_only_ift.py
```

***Tip: Use `ift/ift_implementation.sh` to run these tasks on the CSCS cluster.***

***Open ift/ift_implementation.sh and update the SCRIPT_PATH variable to point to the specific script you want to run (e.g., ift/scrapeCharlotte_to_questions.py) and then submit the job with :***
    
```bash
sbatch ift/ift_implementation.sh
```

### 3. Build the RAG Index ###

Before using RAG, you must generate the embeddings for the IHL rules.
```bash
sbatch RAG/build_rag.sh \
    --rules datasets/rules_with_interpretations.json \
    --outdir RAG/ihl_index \
    --model BAAI/bge-large-en
```

### 4. Evaluate the Model ###

You can evaluate the model on the IHL benchmark in two modes:

* **Standard Evaluation:**
```bash
sbatch evaluation/evaluate.sh --model /path/to/your/checkpoint
```

* **RAG-Augmented Evaluation:**
Injects relevant rules into the prompt to test reasoning with context.
```bash
sbatch evaluation/evaluateRAG.sh \
    --model /path/to/your/checkpoint \
    --index-dir RAG/ihl_index
```







