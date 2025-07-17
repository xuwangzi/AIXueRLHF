export CUDA_VISIBLE_DEVICES=0,1,2,3


accelerate launch --config_file configs/deepspeed_zero3.yaml \
    --main_process_port 29504 \
    src/aixue.py \
    --dataset_name data/aixue_test_data \
    --output_dir /root/group-shared/jrc/ppo-test/models/4gpu_aixue_data_test \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --total_episodes 8 \
    --model_name_or_path /root/group-shared/jrc/base-models/Qwen3-32B \
    --sft_model_path /root/group-shared/jrc/base-models/Qwen3-32B \
    --local_rollout_forward_batch_size 2 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --report_to tensorboard \
    2>&1 | tee -a /root/group-shared/jrc/ppo-test/logs/4gpu_data_test.log