# H200-OPTIMIZED – Qwen2.5-0.5B-Instruct fine-tuning on coding problems
# Expect ~6–8h for a full 3-epoch run with 128k context and massive batch size

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
import wandb

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "0.5b-coding"

# ──────── INITIALIZE WANDB ────────
wandb.init(
    project="Qwen-coding",
    name="qwen2.5-0.5b-coding-h200",
    config={
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_seq_length": 131072,
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 1,
        "learning_rate": 1.5e-4,
        "weight_decay": 0.02,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine_with_restarts",
        "optim": "adamw_torch",
        "r": 64,
        "lora_alpha": 64,
    }
)

# ──────── MAXIMUM H200 SETTINGS ────────
max_seq_length = 131072          # 128K context – Qwen2.5 native with YaRN (Unsloth supports it out of the box)
dtype = torch.bfloat16           # H200 loves bfloat16
load_in_4bit = True              # Still saves a ton of memory, zero accuracy drop on 0.5B

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",           # Unsloth will put everything on the H200
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                                            # H200 can easily handle r=64 or even 128
    lora_alpha=64,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",            # mandatory for 128K
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ──────── DATASET ────────
dataset = load_dataset("open-r1/verifiable-coding-problems-python", split="train")

def format_example(example):
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Solve the given coding problem by writing clean, efficient, and correct code."},
        {"role": "user", "content": example["problem_statement"]},
        {"role": "assistant", "content": example["gold_standard_solution"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=dataset.column_names, num_proc=16)
dataset = dataset.train_test_split(test_size=0.08, seed=3407)  # 92/8 split

print(f"Training samples: {len(dataset['train'])} | Eval samples: {len(dataset['test'])}")

# ──────── H200-MAXED TRAINING ARGS ────────
training_args = TrainingArguments(
    output_dir="./qwen2.5-0.5b-coding-h200",
    per_device_train_batch_size=32,     
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,      
    num_train_epochs=1,                  
    learning_rate=1.5e-4,
    weight_decay=0.02,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_restarts",
    optim="adamw_torch",                 # 8bit is unnecessary on H200 – full adamw is faster & better
    bf16=True,
    fp16=False,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=400,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    run_name="qwen2.5-0.5b-coding-h200",
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    torch_compile=True,                  # torch 2.4+ → ~15–25% faster on H200
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,                       # no need – single examples already hit 128K often
)

# ──────── TRAIN ────────
trainer.train()

# ──────── SAVE FINAL MODELS ────────
model.save_pretrained("./qwen2.5-0.5b-coding-h200-final")
tokenizer.save_pretrained("./qwen2.5-0.5b-coding-h200-final")

# ──────── UPLOAD TO HUGGING FACE HUB ────────
if HF_TOKEN:
    print(f"Uploading model to Hugging Face Hub: {HF_REPO_ID}")
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(
        folder_path="./qwen2.5-0.5b-coding-h200-final",
        repo_id=HF_REPO_ID,
        repo_type="model"
    )
    print("Upload complete!")
else:
    print("HF_TOKEN not found. Skipping upload to Hugging Face Hub.")
