CUDA_LAUNCH_BLOCKING=1

python ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir /root/group-shared/jrc/ppo-test/models/ppo_tldr_size_debug \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --eval_strategy steps \
    --eval_steps 100 \
    --report_to tensorboard \
    2>&1 | tee /root/group-shared/jrc/ppo-test/logs/ppo_tldr_size_debug.log