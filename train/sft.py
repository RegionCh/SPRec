import os
import torch
import re
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
# from utils import find_all_linear_names, print_trainable_parameters
import random
from accelerate import Accelerator

import torch
import bitsandbytes as bnb
import fire


def train(
    # path
    output_dir="",
    base_model ="",
    train_dataset="",
    valid_dataset="",
    train_sample_size:int = 1024,
    resume_from_checkpoint: str = "base_model",  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "",
    wandb_name: str = "",   # the name of the wandb run
    # training hyperparameters
    gradient_accumulation_steps: int = 1,
    batch_size: int = 8,
    num_train_epochs: int = 5,
    learning_rate: float = 2e-5,
    cutoff_len: int = 512,
    eval_step = 0.05,  
    seed=0
):
    os.environ['WANDB_PROJECT'] = wandb_project

    def formatting_prompts_func(examples):
        output_text = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i]
            response = examples["output"][i]

            if len(input_text) >= 2:
                text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                
                ### Instruction:
                {instruction}
                
                ### Input:
                {input_text}
                
                ### Response:
                {response}
                '''
            else:
                text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                
                ### Instruction:
                {instruction}
                
                ### Response:
                {response}
                '''
            output_text.append(text)

        return output_text
    
    train_dataset = load_dataset("json", data_files=train_dataset)
    train_data = train_dataset["train"].shuffle(seed=seed).select(range(train_sample_size))
    val_dataset = load_dataset("json", data_files=valid_dataset)
    val_data = val_dataset["train"].shuffle(seed=seed).select(range(int(train_sample_size/8)))

    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    device_index = Accelerator().process_index
    device_map = {"": device_index}
    #device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  

    if resume_from_checkpoint!="base_model":
        model = PeftModel.from_pretrained(
            model, 
            resume_from_checkpoint, 
            is_trainable=True
        )
    else:
        peft_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=32,
        target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    training_args = SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=20,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=None,
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        max_seq_length=cutoff_len,
        args=training_args
    )

    trainer.train() 
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(output_dir,safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)