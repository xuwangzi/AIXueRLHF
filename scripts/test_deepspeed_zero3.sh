export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


accelerate launch --config_file deepspeed_zero3.yaml \
    --main_process_port 29504 \
    ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir /root/group-shared/jrc/ppo-test/models/32B-8gpu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 100000\
    --model_name_or_path /root/group-shared/jrc/base-models/Qwen3-32B \
    --sft_model_path /root/group-shared/jrc/base-models/Qwen3-32B \
    --reward_model_path /root/group-shared/jrc/base-models/Skywork-Reward-V2-Qwen3-0.6B \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 100 \
    --report_to tensorboard \
    --response_length 128