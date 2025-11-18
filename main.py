import sys
import pandas as pd
import json
import time
from peft import PeftModel
import argparse


from llm.ask_llm import load_llm_model, get_response_llm
from parser.sql_parser import extract_sql_from_text
from read_data.read_dev import load_spider_filtered_ddl, load_ddl, load_table_ddl


lora_model_path = "./sql-llama-lora" 

# model_name = 'codellama/CodeLlama-7b-Instruct-hf'
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "meta-llama/Llama-3.2-3B-Instruct"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--output",  default="meta_predicted_sql.txt")
    ap.add_argument("--backend", choices=["hf"], default="hf")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=600)

    return ap.parse_args()
#--------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------



if len(sys.argv)>1 and sys.argv[1] == "--dataset":
    DATASET_TABLE = sys.argv[2]+"tables.json"
    DATASET_DEV = sys.argv[2]+"dev.json"


if __name__ == '__main__':
    args = parse_args()

    dev_dataset, db_ids, questions = load_ddl(DATASET_DEV, DATASET_TABLE)

    base_model, tokenizer=None,None

    
    base_model, tokenizer = load_llm_model(base_model, tokenizer, model_name)
    

    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    
    predictions = []
    
    # 시작 시간 기록
    start_time = time.time()
    
    # dev_dataset 첫 번째 샘플부터 1034번째 샘플까지
    for i in range(len(dev_dataset)):
        ex = dev_dataset[i]
        question = questions[i]
        db_id = db_ids[i]
        
        print(f"#query_{i}: {question}")
        # print(ex['input'])
        
        # 프롬프트 생성 (instruction-style)
        prompt_text = (
            # f"Given the following database schema:\n{ex['schema']}\nAnswer the following:{ex['input']}\nanswer:" # pure-KD
            
            # f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Output:\n"
            # f"### Instruction:\n{ex['instruction']}\n\n###Database Schema:\n{ex['whole_schema']}\n\n### Input:\n{ex['input']}\n\n### Output:"
            # f"### Instruction:\n{ex['instruction']}\n\n###Database Schema:\n{ex['schema']}\n\n### Input:\n{ex['input']}\n\n### Output:"
            f"### Instruction:\n{ex['instruction']}\n\n###Database Schema:\n{ex['schema']}\n\n### Input:\n{ex['input']}\n\n### Hint:\n{ex['hint']}\n\n### Output:"
            
            # f"### Instruction:\n{ex['instruction']}\n\n###Database Schema:\n{ex['schema']}\n\n### Input:\n{ex['input']}\n\n### AST:\n{ex['ast']}\n\n### Output:"
        )
        # print(prompt_text)

        # SQL 생성
        llm_answer = get_response_llm(model, tokenizer, prompt_text, model_name=model_name, max_tokens=600, temperature=0.0)
        sql_pred = extract_sql_from_text(llm_answer) # sql문만 추출
        print(sql_pred)
        # print(llm_answer)
        print()
        predictions.append(f'{pred_sql}\t{db_id}')

    # 종료 시간 기록
    end_time = time.time()

    with open("output/dev_predictions.sql", "w") as f:
        f.write('\n'.join(predictions))

    # with open("output/dev_predictions.sql", "w") as f:
    #     json.dump(predictions, f, indent=2, ensure_ascii=False)

    
    # 걸린 시간 출력
    print(f"elapsed_time: {end_time - start_time:.4f}s")
    # 결과를 파일에 저장
    with open("output/elasped_time.txt", "w") as f:
        f.write(f"elasped_time: {end_time - start_time:.4f}s\n")
    
   
