DATA_FILE='train.pkl'
OUTPUT_PATH='./tuned_models'
ARCH='llama3'
MODEL_WEIGHTS='../model_weights'
MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
DATA_PATH="./data/LaMP_4/processed"

CUDA_VISIBLE_DEVICES=0 python ./sft.py \
    --arch "${ARCH}" \
    --modelweight "${MODEL_WEIGHTS}" \
    --model_name "${MODEL_NAME}" \
    --model_out_root "${OUTPUT_PATH}" \
    --train_data_root "${DATA_PATH}" \
    --data_name "train.pkl" \
    --tuned_model_file "adapters_${DATA_FILE}" \
    --data_file lamp