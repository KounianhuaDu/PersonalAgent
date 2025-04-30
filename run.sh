MODEL_PATH="../model_weights/Meta-Llama-3.1-8B-Instruct"
for i in 0.4;
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --model "$MODEL_PATH" \
        --model_name_or_path "$MODEL_PATH" \
        --prune_method wanda_owl_search \
        --graph_string "W:(ABSLOG)-(VAR)-(ATAN)-(7)" \
        --sparsity_ratio "$i" \
        --sparsity_type unstructured \
        --save save_test \
        --dataset_name lamp \
        --eval_dataset lamp \
        --nsamples 128 \
        # --resume_from_checkpoint "$MODEL_PATH"

    # CUDA_VISIBLE_DEVICES=0 python main.py \
    #     --model "$MODEL_PATH" \
    #     --model_name_or_path "$MODEL_PATH" \
    #     --prune_method wanda_owl_search \
    #     --graph_string "W:(ABSLOG)-(VAR)-(ATAN)-(7)" \
    #     --sparsity_ratio "$i" \
    #     --sparsity_type unstructured 
    #     --save save_test \
    #     --dataset_name xlam \
    #     --nsamples 512
done
        