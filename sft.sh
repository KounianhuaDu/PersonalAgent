DATA_FILE='train.pkl'
OUTPUT_PATH='./tuned_models'
ARCH='llama3'
MODEL_WEIGHTS='../model_weights'
MODEL_NAME="Llama-3.2-3B-Instruct"
DATA_PATH="./data/LaMP_4/processed"

CUDA_VISIBLE_DEVICES=0 python ./cloud_tuning.py \
    --model_name "${MODEL_NAME}" \
    --tuned_model_file "cloud_0.1" \
