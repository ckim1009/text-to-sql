import torch
import gc
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# 사용자 정의 모듈
from src.read_train import load_ddl


# =========================
# 0. 공통 데이터셋 로드
# =========================
SPIDER_PATH = ""

try:
    train_dataset = load_ddl(SPIDER_PATH, 'spider') 
except NameError:
    print("Warning: load_ddl function not found.")
    train_dataset = None 

# =========================
# 1. 학습 설정 정의 (모델별 모듈 지정)
# =========================
# (모델명, 저장경로, 타겟모듈리스트)
method_name = 'URFine'
models_schedule = [
    {
        "name": "microsoft/Phi-4-mini-Instruct", 
        "output": f"./{method_name}-phi-4-lora",
        # lm_head 제외, 확인된 Projection Layer 사용
        "target_modules": ['qkv_proj', 'down_proj', 'gate_up_proj', 'o_proj']
    },
    {
        "name": "meta-llama/Llama-3.2-3B-Instruct", 
        "output": f"./{method_name}-llama-3.2-3b-lora",
        "target_modules": None
    },
]

# =========================
# 2. 학습 실행 함수
# =========================
def train_and_save_lora(config_info, dataset):
    model_name = config_info["name"]
    output_dir = config_info["output"]
    target_modules = config_info["target_modules"]

    print(f"\n\n{'='*50}")
    print(f"Starting training pipeline for: {model_name}")
    
    if target_modules:
        print(f"Custom Target Modules: {target_modules}")
    else:
        print("Target Modules: Default (None specified)")
        
    print(f"{'='*50}\n")

    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Config 설정 (FlashAttention OFF)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_cache = False 
    if hasattr(config, "_flash_attn_2_enabled"):
        config._flash_attn_2_enabled = False
    if hasattr(config, "use_flash_attention"):
        config.use_flash_attention = False

    # 3. 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True
    )

    # 4. LoRA 설정 (조건문 적용)
    if target_modules is not None:
        # Phi-4, Qwen용: 지정된 모듈 사용
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )
    else:
        # Llama용: target_modules 지정 안 함 (PEFT 라이브러리 기본값 사용)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128,
            lora_alpha=32,
            lora_dropout=0.1
            # target_modules 파라미터 생략
        )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        num_train_epochs=2,
        save_strategy="epoch",
        logging_steps=50,
        bf16=True,
        optim="paged_adamw_32bit",
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # 6. Trainer 초기화
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=lambda ex: f"###Database Schema:\n{ex['schema']}\n### {ex['hint']}\n### Question: {ex['input']}\n### SQL: {ex['output']}"
    )

    # 7. 학습 시작
    trainer.train()

    # 8. 저장
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Successfully saved {model_name} to {output_dir}")

    # 9. 메모리 정리
    del model
    del trainer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory cleared after {model_name}.\n")


# =========================
# 3. 순차 실행 루프
# =========================
for info in models_schedule:
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        train_and_save_lora(info, train_dataset)
    except Exception as e:
        print(f"!!! Failed to train {info['name']} !!!")
        print(f"Error: {e}")
        continue

print("\nAll scheduled training jobs finished.")
