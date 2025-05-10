MODEL_PATH="../model_weights/Qwen1.5-14B-Chat"
for i in 0.3;
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
        # --resume_from_checkpoint ../model_weights/Qwen1.5-14B-Chat
        
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
        