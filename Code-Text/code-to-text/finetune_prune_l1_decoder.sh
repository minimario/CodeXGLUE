export CUDA_VISIBLE_DEVICES=7
cd code
lang=ruby #programming language
lr=5e-5
batch_size=8
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang\_pruned_l1_decoder
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10
pretrained_model=microsoft/codebert-base

python run.py \
    --do_train \
    --do_eval \
    --prune \
    --model_type roberta \
    --model_name_or_path $pretrained_model \
    --train_filename $train_file \
    --dev_filename $dev_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --learning_rate $lr \
    --prune_method l1 \
    --pruned_layer decoder
    --num_train_epochs $epochs 2>&1 | tee pruned_logs/finetune_l1_pruning.log &
