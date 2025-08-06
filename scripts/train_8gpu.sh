export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_DEBUG=INFO

accelerate launch --config_file configs/deepspeed_zero3.yaml \
    --main_process_port 29504 \
    src/aixue.py \
    --dataset_name data/aixue_train_1024_prompt \
    --output_dir /root/group-shared/jrc/ppo-test/models/train_8gpu_wandb \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --total_episodes 1024 \
    --num_ppo_epochs 4 \
    --model_name_or_path /root/group-shared/jrc/base-models/Qwen3-32B \
    --sft_model_path /root/group-shared/jrc/base-models/Qwen3-32B \
    --local_rollout_forward_batch_size 4 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --report_to wandb tensorboard \
    --run_name train_8gpu \
    --save_safetensors true \
    --attn_implementation flash_attention_2 \
    --use_liger_loss false \
    --kl_estimator k3 \
    --temperature 0.6 \
    --whiten_advantages true \
    --kl_coef 0.2 \
    2>&1 | tee -a /root/group-shared/jrc/ppo-test/logs/train_8gpu_wandb.log