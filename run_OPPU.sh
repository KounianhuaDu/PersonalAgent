export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="../model_weights/Llama-3.2-3B-Instruct"
MODEL_NAME="Llama-3.2-3B-Instruct"
K=0
TASK_NAME="news_headline"

cd ../OPPU-main
python ./task_LoRA.py --k "$K" --task_name "$TASK_NAME" --model_name "$MODEL_PATH"

python ./OPPU.py \
    --k "$K" \
    --task_name "$TASK_NAME" \
    --model_name "$MODEL_PATH" \
    --task_lora "./ckpt/${TASK_NAME}/k${K}-${TASK_NAME}-${MODEL_NAME}-task_LoRA_ckpt"

cd ../PersonalAgent