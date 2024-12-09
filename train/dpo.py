import os
import torch
import re
import random

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import find_all_linear_names, print_trainable_parameters
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch.nn.functional as F
import torch
import bitsandbytes as bnb
from accelerate import Accelerator
import fire


def main(
    train_dataset = "",
    val_dataset = "",
    load_8bit: bool = True,
    base_model: str = "",
    num_train_epochs: int = 1,
    gradient_accumulation_steps: int = 4,
    output_dir: str = "",
    wandb_project: str = "self_play",
    wandb_name: str = "",   # the name of the wandb run
    batch_size:int = 2,
    num_epochs:int = 1,
    alpha:float = 1.5,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    eval_step = 0.05,  
    resume_from_checkpoint:bool = False,
    seed = 99
):

    os.environ['WANDB_PROJECT'] = wandb_project
    
    train_dataset = load_dataset("json", data_files=train_dataset)
    train_data = train_dataset["train"].shuffle(seed=seed)
    val_dataset = load_dataset("json", data_files=val_dataset)
    val_data = val_dataset["train"].shuffle(seed=seed)

    device_index = Accelerator().process_index
    device_map = {"": device_index}
        
    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    device_index = Accelerator().process_index
    device_map = {"": device_index}

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

    model_ref = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=bnb_config
    )
    
    if resume_from_checkpoint:
        reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
    else: 
        reference_model = model_ref


    training_args = DPOConfig(
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
    )

    dpo_trainer = DPOTrainer(
        model,
        reference_model,
        args=training_args,
        beta=0.1,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )


    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)


    print("DPO training is done")

if __name__ == "__main__":
    fire.Fire(main)
