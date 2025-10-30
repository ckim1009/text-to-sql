from transformers import StoppingCriteria, StoppingCriteriaList

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch 
import time


class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_sequences):
        super().__init__()
        # stop_sequences: list[list[int]] (각 시퀀스의 토큰 id)
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # 배치 1 가정
        generated = input_ids[0].tolist()
        for seq in self.stop_sequences:
            L = len(seq)
            if L > 0 and generated[-L:] == seq:
                return True
        return False




def load_llm_model(model, tokenizer, model_name):

    if model is None or tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        quantization_config = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype="auto"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            # device_map="auto",
            device_map={"": device},
            trust_remote_code=True,
            quantization_config=quantization_config,
            attn_implementation="eager",
        )
        model.eval()

        print("is_loaded_in_4bit:", getattr(model, "is_loaded_in_4bit", False))
    return model, tokenizer

def get_response_llm(model, tokenizer, prompt, model_name, max_tokens=600, temperature=0.0, stop=None):
    model, tokenizer = load_llm_model(model, tokenizer, model_name)

    while True:
        try:
            if isinstance(prompt, list) and isinstance(prompt[0], dict):
                prompt = "\n".join([msg["content"] for msg in prompt if "content" in msg])
            if not isinstance(prompt, str):
                raise ValueError(f"Prompt must be a string, got {type(prompt)}")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                # temperature=temperature,
                do_sample=False,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):].strip()
            if stop:
                for token in stop:
                    if token in decoded:
                        decoded = decoded.split(token)[0].strip()
                        break
            return decoded
        except Exception as e:
            print('Retrying due to LLM error:', repr(e))
            time.sleep(10)
