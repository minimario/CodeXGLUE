export CUDA_VISIBLE_DEVICES=7
cd code
lang=ruby
lr=5e-5
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang
batch_size=16
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
# test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
test_model=/scratch/gua/Documents2/CodeXGLUE/Code-Text/code-to-text/code/model/ruby_nov27_pruned_best_l1/checkpoint-0/pytorch_model.bin
# test_model=/scratch/gua/Documents2/CodeXGLUE/Code-Text/code-to-text/code/model/ruby_nov27_pruned_best_random/checkpoint-0/pytorch_model.bin

python run.py \
    --do_test \
    --prune \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --load_model_path $test_model \
    --dev_filename $dev_file \
    --test_filename $test_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --beam_size $beam_size \
    --eval_batch_size $batch_size
