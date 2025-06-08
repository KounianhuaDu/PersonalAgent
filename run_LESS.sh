cd ../LESS

source less/scripts/train/base_training_args.sh

data_dir=../PersonalAgent/data
model_path=../model_weights/Llama-3.2-3B-Instruct
percentage=0.1
data_seed=0
job_name=llama3-3B-p${percentage}-lora-seed${data_seed}

output_dir=./out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

train_files="$data_dir/LaMP_4/processed/remain_train.pkl"
    

# use fsdp for large models
if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
    elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
fi

# wramup lora
training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--train_files $train_files"

# --train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args" # yeild checkpoints in line 39, used to compute gradient

CKPT=60
model=./out/llama3-3B-p0.05-lora/checkpoint-${CKPT} # path to model
dims="8192" # dimension of projection, can be a list
gradient_type="adam"
output_path=./grads/llama3-3B-p0.05-lora/lamp-ckpt${CKPT}-${gradient_type} # path to output

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi
# get gradient for train data
CUDA_LAUNCH_BLOCKING=1 python3 -m less.data_selection.get_info \
--train_file $train_files \
--info_type grads \
--model_path $model \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type $gradient_type

task=lamp # tydiqa, mmlu
output_path=./grads/llama3-3B-p0.05-lora/${task}-ckpt${CKPT}-sgd # path to output

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi
# get gradients for val data
# I get this data from profiles of remain_train.pkl which does not overlap with train data
python3 -m less.data_selection.get_info \
--task $task \
--train_file $train_files \
--info_type grads \
--model_path $model \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type sgd \
--data_dir $data_dir

#!/bin/bash
OPTIM=$gradient_type
gradient_path=./grads/llama3-3B-p0.05-lora/{}-ckpt{}-${OPTIM}_train/dim${DIM}
train_file_names="lamp_${OPTIM}"
ckpts="60"
checkpoint_weights="1"

validation_gradient_path=./grads/llama3-3B-p0.05-lora/{}-ckpt{}-sgd/dim${DIM}
output_path=./selected_data

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

# calculate the influence score for each training data point
python3 -m less.data_selection.matching \
--gradient_path $gradient_path \
--train_file_names $train_file_names \
--ckpts $ckpts \
--checkpoint_weights $checkpoint_weights \
--validation_gradient_path $validation_gradient_path \
--target_task_names $task\
--output_path $output_path

# select data with the highest influence score
python3 -m less.data_selection.write_selected_data \
--target_task_names ${target_task_names} \
--train_file_names ${train_file_names} \
--train_files ${train_files} \
--output_path $output_path \
--percentage 0.1

PERCENTAGE=0.1
train_files=./selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}_lamp_${OPTIM}.jsonl
job_name=llama3-3B-less-p${PERCENTAGE}-lora-${OPTIM}

output_dir=./out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

# use fsdp for large models
#if [[ $model_path == "meta-llama/Llama-2-13b-hf" ]]; then
#    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
#    elif [[ $model_path == "mistralai/Mistral-7B-v0.1" ]]; then
#    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
#fi

# final lora
training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--train_files $train_files" \
# --train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

echo "$header $training_args"
eval "$header" "$training_args"

cd ../PersonalAgent