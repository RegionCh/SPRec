import re
import json
import sys
import fire
import gradio as gr
import numpy as np
import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer
import random
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig,AutoTokenizer
from transformers import LlamaForCausalLM
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(
    json_file : str  = "",
    result_json_dpo_data_train: str = "",
    result_json_dpo_data_valid: str = "",
    result_json_sft_data_train: str = "",
    result_json_sft_data_valid: str = "",
    base_model: str = "",
    lora_weights: str = "",
    batch_size:int = 4,
    train_sample_size:int = 1024,
    valid_sample_size:int = 128,
    load_8bit: bool = False,
    random_neg: bool = False,
):
    sample_size = train_sample_size + valid_sample_size + 500 # In case fail to generate

    # generate responses from model
    tokenizer =  AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    load_8bit = False
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    tokenizer.padding_side = "left"

    model.eval()

    #emb_model = SentenceTransformer('/data/chenruijun/code/models/paraphrase-MiniLM-L3-v2')

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                pad_token_id = tokenizer.eos_token_id
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs
    
    outputs = []
    tokenizer.pad_token_id = tokenizer.eos_token_id
    from tqdm import tqdm
    with open(json_file, 'r') as f:
        test_data = json.load(f)
        test_data = random.sample(test_data, sample_size)
        sft_train_data = test_data[:train_sample_size]
        sft_valid_data = test_data[train_sample_size:train_sample_size + valid_sample_size]
        with open(result_json_sft_data_train, 'w') as f:
            for item in sft_train_data:
                json.dump(item, f) 
                f.write('\n') 
        with open(result_json_sft_data_valid, 'w') as f:
            for item in sft_valid_data:
                json.dump(item, f) 
                f.write('\n')    

        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        def batch(list, batch_size=batch_size):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions, inputs = batch
            output = evaluate(instructions, inputs)
            outputs = outputs + output
            
        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['predict'] = outputs[i]

    dpo_data = []
    count = 0
    print(f"Length of test_data : {len(test_data)}")
    for data in test_data:
        dpo_case = {}
        dpo_case['prompt'] = data['instruction'] + data['input']
        dpo_case['chosen'] = data['output']
        pattern = r'"(.*?)"'
        item_names = re.findall(pattern, data['predict'][0])
        formatted_item_names = [f'\"{item}\"' for item in item_names]
        if(len(formatted_item_names)==0):
            continue
        dpo_case['rejected'] = formatted_item_names[0]+"\n"
        dpo_data.append(dpo_case)
        count += 1
        if count == train_sample_size + valid_sample_size:
            break
    random.shuffle(dpo_data)
    dpo_train_data = dpo_data[:train_sample_size]
    dpo_valid_data = dpo_data[train_sample_size:]


    with open(result_json_dpo_data_train, 'w') as f:
        for item in dpo_train_data:
            json.dump(item, f)  # 将字典转换为 JSON 格式写入文件
            f.write('\n')  # 每个 JSON 对象之间用换行分隔

    with open(result_json_dpo_data_valid, 'w') as f:
        for item in dpo_valid_data:
            json.dump(item, f)  # 将字典转换为 JSON 格式写入文件
            f.write('\n')  # 每个 JSON 对象之间用换行分隔


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)
