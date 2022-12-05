export CUDA_VISIBLE_DEVICES=5 
cd code
lang=ruby #programming language
lr=5e-5
batch_size=8
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang\_pruned_random_structured
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10 
pretrained_model=microsoft/codebert-base
load_model_path=model/ruby/checkpoint-best-bleu/pytorch_model.bin

python run.py \
    --do_train \
    --do_eval \
    --prune \
    --model_type roberta \
    --load_model_path $load_model_path \
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
    --prune_method random_structured \
    --num_train_epochs $epochs 2>&1 | tee pruned_logs/finetune_random_pruning_structured.log &
