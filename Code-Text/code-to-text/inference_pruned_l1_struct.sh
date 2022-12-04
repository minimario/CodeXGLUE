cd code
lang=ruby
lr=5e-5
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/ruby_pruned_l1_structured
batch_size=16
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl

# device=(1 1 3 3 4 4 6 6 7 7)

for i in 1 2 3 4 5 6
do
    # device_id=${device[$i]}
    checkpoint_id=$((i))
    test_model=$output_dir/checkpoint-0-while-$checkpoint_id/pytorch_model.bin
    # export CUDA_VISIBLE_DEVICES=$device_id
    python3 run.py \
        --do_test \
        --prune \
        --model_type roberta \
        --model_name_or_path  microsoft/codebert-base \
        --load_model_path $test_model \
        --dev_filename $dev_file \
        --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size 2>&1 | tee logs/dec4_pruned_best_l1_struct/pruned_$checkpoint_id.log &
done