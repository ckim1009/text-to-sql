import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# import bitsandbytes as bnb

import os
import sqlite3
from datasets import concatenate_datasets

from src.read_train import load_ddl


# GPU 0번 디바이스 고정
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
# torch.cuda.set_device(4)
# device = torch.device("cuda:4")

output_dir = ""
SPIDER_PATH = ""


train_dataset = load_ddl(SPIDER_PATH ,'spider') # sql schema hint(column)

# train_dataset1 = load_ddl(SPIDER_PATH ,'spider')
# train_dataset2 = load_ddl(SPIDER_PATH ,'spider-syn')
# train_datset = concatenate_datasets([train_dataset1, train_dataset2])

# =========================
# 2. 모델 & 토크나이저 불러오기
# =========================
# model_name = "microsoft/Phi-4-mini-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "google/gemma-3-4b-it"


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# # Config 불러와서 FlashAttention 비활성화
# config = AutoConfig.from_pretrained(model_name)
# config.use_flash_attention = False

# # 4bit 양자화 설정
# bnb_config = bnb.BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )


# Config 설정으로 FlashAttention 비활성화
config = AutoConfig.from_pretrained(model_name)

# FlashAttention 관련 설정들을 명시적으로 비활성화
config.use_cache = False  # LoRA 학습에서는 cache 비활성화
if hasattr(config, "_flash_attn_2_enabled"):
    config._flash_attn_2_enabled = False
if hasattr(config, "use_flash_attention"):
    config.use_flash_attention = False
    
# 모델 로드 시 attention implementation 명시
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    # device_map={"": device},  # 모든 레이어를 GPU 0번에 할당
    torch_dtype=torch.float16,
    attn_implementation="eager",  # FlashAttention 대신 eager implementation 사용
    trust_remote_code=True
)

# =========================
# 3. LoRA 설정
# =========================
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1
# )
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,
    lora_alpha=32,
    lora_dropout=0.1
)


model = get_peft_model(model, lora_config)

# =========================
# 4. Trainer 세팅
# =========================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,# 1e-4,
    num_train_epochs=2,
    save_strategy="epoch",
    logging_steps=50,
    bf16=True,
    optim="paged_adamw_32bit",
    save_total_limit=3,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    formatting_func=lambda ex: f"### Instruction:\n{ex['instruction']}\n\n###Database Schema:\n{ex['schema']}\n\n### Hint:\n{ex['hint']}\n\n### Input:\n{ex['input']}\n\n### Output:\n{ex['output']}"

)

# =========================
# 5. 학습 시작
# =========================
trainer.train()


# 학습 종료 후
trainer.model.save_pretrained(output_dir)  # LoRA weight만 저장
