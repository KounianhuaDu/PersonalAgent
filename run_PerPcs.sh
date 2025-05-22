export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="../model_weights/Llama-3.2-3B-Instruct"
TASK_NAME="news_headline"

cd ../Per-Pcs-main
python ./task_LoRA.py \
    --task_name "$TASK_NAME" \
    --llama_model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH"

# Clustering training data. Only run once for each dataset
# python ../Per-Pcs-main/anchor_selection/history_anchor.py \
#     --candidate_path  "../../PersonalAgent/data/LaMP_4/processed/train.pkl" \
#     --task_name "$TASK_NAME" \
#     --k 50

python ./train_anchor_PEFT.py \
    --task_name "$TASK_NAME" \
    --llama_model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --lora_ckpt "./output/${TASK_NAME}/task-base_LLM/lora_ckpt.pt" \
    --output_dir "./output/${TASK_NAME}/Anchor_PEFT/LoRA"

python train_anchor_gate.py \
    --task_name "$TASK_NAME" \
    --llama_model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --lora_ckpt "./output/${TASK_NAME}/task-base_LLM/lora_ckpt.pt" \
    --anchor_path "./output/${TASK_NAME}/Anchor_PEFT/LoRA" \
    --anchor_idx_path "./anchor_selection/${TASK_NAME}/anchor_user_idx.pt" \
    --output_dir "./output/${TASK_NAME}/Anchor_PEFT/gate"

python lora_composition.py \
    --task_name "$TASK_NAME" \
    --llama_model_path "$MODEL_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --output_dir "./output/${TASK_NAME}/LoRA-Composition" \
    --lora_ckpt "./output/${TASK_NAME}/task-base_LLM/lora_ckpt.pt" \
    --gate_dir "./output/${TASK_NAME}/Anchor_PEFT/gate"\
    --anchor_dir "./output/${TASK_NAME}/Anchor_PEFT/LoRA" \
    # --topk 3 \
    # --recent_k 2000 \
    # --share_ratio 1 \
    # --agg_temperature 1

cd ../PersonalAgent