device=0

mode=safety

data_path=../../pa_back/caa_data
model_name=llama-3.1
data_names=(
    caa_python_LaMP_4_0.15
)
model_name_or_path=../../model_weights/fix/Meta-Llama-3.1-8B-Instruct # replace ./model/gemma-2-9b-it with your own model path
layers=(20)
for eval_data_name in ${data_names[@]}; do
    data_name=${eval_data_name}
    # log_path=${data_path}/${data_name}/caa_vector_it/${model_name}_${mode}/logs/${model_name}_${data_name}.log
    # echo "Starting the script..."
    #  # 创建日志目录
    # log_dir=$(dirname ${log_path})
    # echo "Log directory: ${log_dir}"
    # if [ ! -d "${log_dir}" ]; then
    #     echo "Creating log directory..."
    #     mkdir -p "${log_dir}"
    # else
    #     echo "Log directory already exists."
    # fi

    # # 打印生成的日志路径
    # echo "Log path: ${log_path}"

    echo "Running Python script for ${data_name}..."

    # CUDA_VISIBLE_DEVICES=${device} python ./generate_vectors.py \
    #     --mode ${mode} \
    #     --layers ${layers} \
    #     --save_activations \
    #     --model_name ${model_name} \
    #     --data_path ${data_path} \
    #     --data_name ${data_name} \
    #     --data_type safety \
    #     --model_name_or_path ${model_name_or_path} 
    #     # > ${log_path} 2>&1 

done
echo "Script execution completed."

data_path=../../pa_back/data
layer_num=${#layers[@]}
data_names=(
    ../../pa_back/caa_data
)
form='raw'

for data_name in ${data_names[@]}; do
    for eval_data_name in LaMP_4; do
        for ((i=0; i<${layer_num}; i++)); do
            layer=${layers[$i]}
            # log_path=./results/${data_name}/${model_name}_results_${mode}/logs/main/caa/eval_${eval_data_name}/${model_name}_steer${data_name}_caa__layer${layer}.result.log

            # log_dir=$(dirname ${log_path})
            # if [ ! -d "${log_dir}" ]; then
            #     mkdir -p "${log_dir}"
            # fi

            output_file=./results/${model_name}_results_${mode}/eval_${eval_data_name}/caa__layer${layer}_${form}.json

            CUDA_VISIBLE_DEVICES=${device} python ./steering_caa.py \
                --mode ${mode} \
                --layers ${layer} \
                --multipliers 1 \
                --eval_data_name ${eval_data_name} \
                --model_name ${model_name} \
                --data_name ${data_name} \
                --data_path ${data_path} \
                --model_name_or_path ${model_name_or_path} \
                --output_file ${output_file} \
                --form ${form}
                # > ${log_path} 2>&1 

        done
    done
done
