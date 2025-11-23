import sys
import time
import os
import torch
import gc
import argparse
import pandas as pd
import json
from peft import PeftModel

# 사용자 정의 모듈
from src.ask_llm import load_llm_model, get_response_llm
from parser.sql_parser import extract_sql_from_text
from src.read_train import load_ddl_dev

# 데이터셋 경로
PATH = ''

# ==========================================
# 1. 추론 스케줄 설정 (모델명, LoRA 경로, 결과 파일 식별자)
# ==========================================
inference_schedule = [
    {
        "base_model": "microsoft/Phi-4-mini-Instruct",
        "lora_path": "./pure-kd-phi-4-lora",
        "output_id": "phi-4"
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "lora_path": "./pure-kd-llama-3.2-3b-lora",
        "output_id": "llama-3b"
    }
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    # output 인자는 루프 내에서 자동으로 생성되므로 기본값은 무시됩니다.
    ap.add_argument("--backend", choices=["hf"], default="hf")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    return ap.parse_args()

def run_inference_for_model(config, dev_dataset, questions, db_ids):
    base_model_name = config["base_model"]
    lora_path = config["lora_path"]
    output_id = config["output_id"]
    
    print(f"\n\n{'='*50}")
    print(f"Starting Inference for: {base_model_name}")
    print(f"LoRA Path: {lora_path}")
    print(f"{'='*50}")

    # 1. 모델 및 토크나이저 로드
    # load_llm_model 함수 내부 구현에 따라 base_model만 로드된다고 가정
    # (기존 코드: base_model, tokenizer = load_llm_model(None, None, model_name))
    try:
        base_model, tokenizer = load_llm_model(None, None, base_model_name)
    except Exception as e:
        print(f"Error loading base model {base_model_name}: {e}")
        return

    # 2. LoRA 어댑터 병합
    print(f"Loading LoRA adapter from {lora_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
    except Exception as e:
        print(f"Error loading LoRA adapter (Check if path exists): {e}")
        # LoRA 로드 실패 시 베이스 모델 메모리 해제 후 리턴
        del base_model
        del tokenizer
        return

    predictions = []
    start_time = time.time()

    # 3. 추론 루프
    total_samples = len(dev_dataset)
    for i in range(total_samples):
        ex = dev_dataset[i]
        question = questions[i]
        db_id = db_ids[i]

        # 진행 상황 출력
        print(f"[{output_id}] #query_{i}/{total_samples}: {question}")

        # 프롬프트 생성 (기존 코드의 formatting 유지)
        prompt_text = f"### Instruction:\n{ex['instruction']}\n\n###Database Schema:\n{ex['whole_schema']}\n\n### Input:\n{ex['input']}\n\n### Output:"
            

        # LLM 생성
        # get_response_llm 내부에서 model.generate 호출
        llm_answer = get_response_llm(
            model, 
            tokenizer, 
            prompt_text, 
            model_name=base_model_name, 
            max_tokens=400, 
            temperature=0.0
        )
        
        # SQL 추출
        sql_pred = extract_sql_from_text(llm_answer)
        print(f"Pred: {sql_pred}\n")
        
        predictions.append(f'{sql_pred}\t{db_id}')

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 4. 결과 저장 (모델별 별도 파일)
    if not os.path.exists("output"):
        os.makedirs("output")

    # SQL 결과 저장
    output_sql_file = f"output/full_dev_predictions_{output_id}.sql"
    with open(output_sql_file, "w") as f:
        f.write('\n'.join(predictions))

    # 시간 기록 저장
    output_time_file = f"output/full_elapsed_time_{output_id}.txt"
    with open(output_time_file, "w") as f:
        f.write(f"model: {base_model_name}\n")
        f.write(f"lora: {lora_path}\n")
        f.write(f"elapsed_time: {elapsed_time:.4f}s\n")
        f.write(f"avg_time_per_query: {elapsed_time/total_samples:.4f}s\n")

    print(f"Finished {output_id}. Saved to {output_sql_file}")
    
    # 5. 메모리 정리 
    del model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory cleared for {output_id}.\n")


if __name__ == '__main__':
    args = parse_args()

    # 데이터셋 로드는 한 번만 수행
    print("Loading dataset...")
    if len(sys.argv) > 1 and sys.argv[1] == "--dataset":
        # sys.argv 사용 시 주의: argparse와 혼용 시 충돌 가능성 있음. 
        # load_ddl_dev 내부 로직에 맞게 경로 설정이 필요하다면 수정.
        pass 
        
    dev_dataset, db_ids, questions = load_ddl_dev(PATH, 'spider')
    print(f"Loaded {len(dev_dataset)} samples.")

    # 스케줄에 있는 모든 모델에 대해 순차적으로 추론 실행
    for config in inference_schedule:
        # 파일 존재 여부 미리 체크 (학습이 안 된 모델 건너뛰기)
        if not os.path.exists(config["lora_path"]):
            print(f"Skipping {config['name']}: LoRA path not found at {config['lora_path']}")
            continue
            
        run_inference_for_model(config, dev_dataset, questions, db_ids)

    print("All inference jobs completed.")
