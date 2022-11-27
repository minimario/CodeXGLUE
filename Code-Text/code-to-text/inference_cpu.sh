export CUDA_VISIBLE_DEVICES=
cd code
lang=ruby
lr=5e-5
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang
batch_size=64
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py \
    --do_test \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --load_model_path $test_model \
    --dev_filename $dev_file \
    --test_filename $test_file \
    --output_dir $output_dir \
    --max_source_length $source_length \
    --max_target_length $target_length \
    --no_cuda \
    --quantize \
    --beam_size $beam_size \
    --eval_batch_size $batch_size
