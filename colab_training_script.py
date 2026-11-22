"""
Google Colab Training Script: Domain-Specific Fine-Tuning of Conversational AI for Enterprise IT
Platform Target: Google Colab with NVIDIA T4 GPU (16 GB VRAM)
Model: Meta-Llama-3-8B-Instruct / Mistral-7B-Instruct-v0.3
Optimization: QLoRA (4-bit NF4 Quantization + LoRA PEFT)
Training Duration: 25 Epochs (As specified in Capstone Research Guidelines)
"""

import os
import torch
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging as hf_logging
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer

# 1. Setup and Environment Verification
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# 2. Configuration Parameters
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # Alternative: "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./results/itsm-llama3-8b-lora-25ep"
DATASET_PATH = "./dataset/itsm_devops_curated_dataset.jsonl"
NUM_EPOCHS = 25
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8  # Effective batch size = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 3. Load and Format Dataset
def load_and_preprocess_dataset(filepath, tokenizer):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    formatted_texts = []
    for item in records:
        messages = item["messages"]
        # Apply Llama-3 instruction chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append({"text": prompt})
    
    # Split into Train (85%) and Validation (15%)
    raw_dataset = Dataset.from_list(formatted_texts)
    split_dataset = raw_dataset.train_test_split(test_size=0.15, seed=42)
    return split_dataset

# 4. BitsAndBytes 4-Bit Quantization Configuration (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 5. Tokenizer Initialization
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 6. Base Model Loading
print("Loading base model in 4-bit precision...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
base_model = prepare_model_for_kbit_training(base_model)

# 7. LoRA Parameter-Efficient Fine-Tuning Setup
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)
peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()

# 8. Prepare Datasets
datasets = load_and_preprocess_dataset(DATASET_PATH, tokenizer)
print(f"Training samples: {len(datasets['train'])}, Validation samples: {len(datasets['test'])}")

# 9. Training Arguments (Capped at 25 Epochs)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

# 10. SFTTrainer Execution
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args
)

print("Starting LoRA fine-tuning for 25 epochs on NVIDIA T4...")
train_result = trainer.train()

# 11. Save Adapter Checkpoints
print(f"Saving fine-tuned adapter weights to {OUTPUT_DIR}/final_adapter")
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

# 12. Inference Demonstration
print("\n=== Demonstrating Fine-Tuned Domain Bot Response ===")
test_prompt = [
    {"role": "system", "content": "You are SysOps-AI, an expert Site Reliability Engineering (SRE) conversational assistant specializing in enterprise Kubernetes infrastructure, Linux kernel internals, and cloud systems."},
    {"role": "user", "content": "Our ingress controller returned HTTP 502 Bad Gateway after deploying new replica set. How do we verify upstream service endpoints?"}
]
formatted_input = tokenizer.apply_chat_template(test_prompt, tokenize=True, return_tensors="pt", add_generation_prompt=True).to("cuda")

with torch.no_grad():
    output_tokens = trainer.model.generate(
        formatted_input,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15
    )

response = tokenizer.decode(output_tokens[0][formatted_input.shape[1]:], skip_special_tokens=True)
print(f"Generated Domain Assistant Response:\n{response}")
