DATA_FILE='04_128'
OUTPUT_PATH='./tuned_models'
ARCH='llama3'
MODEL_WEIGHTS='./'
MODEL_NAME="pruned_model"
DATA_PATH="../PublicEval/LaMP_4/train"

CUDA_VISIBLE_DEVICES=0 python ./sft.py \
    --arch "${ARCH}" \
    --modelweight "${MODEL_WEIGHTS}" \
    --model_name "${MODEL_NAME}" \
    --model_out_root "${OUTPUT_PATH}" \
    --train_data_root "${DATA_PATH}" \
    --data_name "train_questions.json" \
    --tuned_model_file "adapters_${DATA_FILE}" \
    --data_file lamp