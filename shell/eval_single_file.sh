# bash ./shell/eval_single_file.sh 0 2 4 5
base_model=$5
lora_weights=$6
category=$7
# Only change the parameters above if needed
echo -------------------------------------- Evaluation started! --------------------------------------

gpu1=$1; gpu2=$2; gpu3=$3; gpu4=$4
test_json="./data/$category/test.json"
result_json="${lora_weights}/test_result.json"
touch $result_json
CUDA_VISIBLE_DEVICES=$gpu1,$gpu2,$gpu3,$gpu4 python ./eval/inference.py \
    --base_model $base_model \
    --lora_weights $lora_weights \
    --test_data_path $test_json \
    --result_json_data $result_json \
    --num_beams 1
echo Result for model "$lora_weights" is created in $result_json!
eval_result_json="${lora_weights}/eval_top1.json"
CUDA_VISIBLE_DEVICES=$1 python ./eval/evaluate.py \
    --input_dir $result_json \
    --output_dir $eval_result_json \
    --topk 1 \
    --gamma 0 \
    --category $category
echo Metrics for model "$lora_weights" is created in $eval_result_json!
eval_result_json="${lora_weights}/eval_top5.json"
CUDA_VISIBLE_DEVICES=$1 python ./eval/evaluate.py \
    --input_dir $result_json \
    --output_dir $eval_result_json \
    --topk 5 \
    --gamma 0 \
    --category $category
echo Metrics for model "$lora_weights" is created in $eval_result_json!

echo -------------------------------------- Evaluation finished! --------------------------------------