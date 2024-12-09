gpu1=$1; gpu2=$2; gpu3=$3; gpu4=$4; its=$5   
train_sample_size=2048;valid_sample_size=256
base_model="/data/chenruijun/code/models/Llama-3.2-1B-Instruct"
batch_size=4
lr=0.00002
# Only change the parameters above if needed
for category in "MovieLens" "Goodreads" "CDs_and_Vinyl" "Video_Games" "Steam" 
do
    lora_weights="./models/SFT_model_4096/${category}"
    output_dir="./models/SPRec/${category}_${train_sample_size}_${lr}"
    wandb_project="SPRec_${category}_${lr}_${train_sample_size}"
    echo ----------------- Training Parameters -----------------
    echo "GPU: $gpu1,$gpu2,$gpu3,$gpu4"
    echo "Iterations: $its"
    echo "Train Sample Size: $train_sample_size"
    echo "Valid Sample Size: $valid_sample_size"
    echo "Base Model: $base_model"
    echo "LoRA Weights: $lora_weights"
    echo "Category: $category"
    echo "Learning Rate: $lr"

    for ((i=0;i<$its;i++))
    do
        echo ----------------- Iteration$i starts! -----------------
        it_output_dir="${output_dir}/it${i}/"
        dpo_train_data_path="${it_output_dir}/data/dpo_train.jsonl"
        dpo_valid_data_path="${it_output_dir}/data/dpo_valid.jsonl"
        sft_train_data_path="${it_output_dir}/data/sft_train.jsonl"
        sft_valid_data_path="${it_output_dir}/data/sft_valid.jsonl"
        mkdir -p $it_output_dir
        mkdir -p "${it_output_dir}/data"
        touch "${dpo_train_data_path}"
        touch "${dpo_valid_data_path}"
        touch "${sft_train_data_path}"
        touch "${sft_valid_data_path}"
        # Data Generation
        CUDA_VISIBLE_DEVICES=$gpu1,$gpu2,$gpu3,$gpu4 python ./train/data_generate.py \
            --json_file ./data/${category}/train.json \
            --result_json_dpo_data_train $dpo_train_data_path \
            --result_json_dpo_data_valid $dpo_valid_data_path \
            --result_json_sft_data_train $sft_train_data_path \
            --result_json_sft_data_valid $sft_valid_data_path \
            --base_model $base_model \
            --lora_weights $lora_weights \
            --batch_size 64 \
            --train_sample_size $train_sample_size \
            --valid_sample_size $valid_sample_size
        # SFT
        wandb_name="iteration${i}_SFT"
        SFT_path="${it_output_dir}SFT"
        mkdir -p $SFT_path
        CUDA_VISIBLE_DEVICES=$gpu1,$gpu2,$gpu3,$gpu4 python ./train/sft.py \
            --resume_from_checkpoint $lora_weights \
            --output_dir $SFT_path \
            --base_model $base_model \
            --train_dataset $sft_train_data_path \
            --valid_dataset $sft_valid_data_path \
            --train_sample_size $train_sample_size \
            --wandb_project $wandb_project \
            --wandb_name $wandb_name \
            --gradient_accumulation_steps 4 \
            --batch_size $batch_size \
            --num_train_epochs 1 \
            --learning_rate $lr \
            --cutoff_len 512 \
        # Evaluate SFT model
        lora_weights=$SFT_path
        bash ./shell/eval_single_file.sh  $gpu1 $gpu2 $gpu3 $gpu4 \
                                        $base_model \
                                        $lora_weights \
                                        $category
        # DPO
        wandb_name="iteration${i}_DPO"
        DPO_path="${it_output_dir}DPO/"
        mkdir -p $DPO_path
        CUDA_VISIBLE_DEVICES=$gpu1,$gpu2,$gpu3,$gpu4 python ./train/dpo.py \
            --train_dataset $dpo_train_data_path \
            --val_dataset $dpo_valid_data_path \
            --output_dir $DPO_path \
            --base_model $base_model \
            --resume_from_checkpoint $lora_weights \
            --wandb_name $wandb_name \
            --wandb_project $wandb_project \
            --batch_size 2 \
            --gradient_accumulation_steps 4 \
            --learning_rate $lr \
            --cutoff_len 512 \
            --num_epochs 1 
        # Evaluate DPO model
        lora_weights=$DPO_path
        bash ./shell/eval_single_file.sh  $gpu1 $gpu2 $gpu3 $gpu4 \
                                        $base_model \
                                        $lora_weights \
                                        $category 
        
    done
    echo SPRec for category ${category} has successfully completed!
done