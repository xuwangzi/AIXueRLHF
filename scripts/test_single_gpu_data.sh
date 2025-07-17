CUDA_LAUNCH_BLOCKING=1

python aixue.py \
    --dataset_name data/aixue_test.parquet \
    --learning_rate 3e-6 \
    --output_dir /root/group-shared/jrc/ppo-test/models/aixue_data_test \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 4 \
    --model_name_or_path /root/group-shared/models/base_models/Qwen3-0.6B \
    --sft_model_path /root/group-shared/models/base_models/Qwen3-0.6B \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --report_to tensorboard \
    2>&1 | tee /root/group-shared/jrc/ppo-test/logs/aixue_data_test.log