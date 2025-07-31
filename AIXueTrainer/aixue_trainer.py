# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import time
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

from trl.core import masked_mean, masked_whiten
from trl.import_utils import is_liger_kernel_available
from trl.models import create_reference_model
from trl.trainer.utils import (
    OnlineTrainerState,
    disable_dropout_in_model,
    first_true_indices,
    forward,
    prepare_deepspeed,
    selective_log_softmax,
    truncate_response,
)

import deepspeed
from tqdm import tqdm

if is_liger_kernel_available():
    from AIXueTrainer.aixue_loss import LigerFusedLinearAIXueLoss

from AIXueTrainer.aixue_config import AIXueConfig

INVALID_LOGPROB = 1.0

class PromptResponseDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        max_seq_length = max(len(item["input_ids"]) for item in batch)
        
        padded_input_ids_batch = []
        for item in batch:
            padded_seq = [self.tokenizer.pad_token_id] * (max_seq_length - len(item["input_ids"])) + item["input_ids"]
            padded_input_ids_batch.append(padded_seq)
        
        max_seq_length = max(len(item["response_ids"]) for item in batch)
        
        padded_response_ids_batch = []
        for item in batch:
            padded_seq = item["response_ids"] + [self.tokenizer.pad_token_id] * (max_seq_length - len(item["response_ids"]))
            padded_response_ids_batch.append(padded_seq)
        
        return {
            "input_ids": torch.tensor(padded_input_ids_batch, dtype=torch.long),
            "response_ids": torch.tensor(padded_response_ids_batch, dtype=torch.long),
            "reward": torch.tensor([item["reward"] for item in batch], dtype=torch.float)
        }

class AIXueTrainer(Trainer):
    _tag_names = ["trl", "aixue"]

    def __init__(
        self,
        args: AIXueConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ], # tokenizer
        model: nn.Module,
        ref_model: Optional[nn.Module],
        train_dataset: Dataset,
        data_collator: Optional[PromptResponseDataCollator] = None, # padding
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None, # 根据Trainer状态进行控制
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model

        # Define the collator if not provided
        if data_collator is None:
            data_collator = PromptResponseDataCollator(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # Check that the kl estimator is valid
        if self.args.kl_estimator not in {"k1", "k3"}: # KL散度估计
            raise ValueError(
                "kl_estimator must be either 'k1' (straightforward, unbiased) or 'k3' (lower variance, unbiased, "
                "appears to be a strictly better estimator). See "
                "[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) for details."
            )


        if ref_model:
            self.ref_model = ref_model
        else:
            self.ref_model = create_reference_model(self.policy_model)

        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size
        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.ref_model]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = self.policy_model
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None # 超参数搜索
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names) # [trl,aixue]

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
            # 不drop如何做？
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        if self.is_deepspeed_enabled:
            if self.ref_model is None:
                raise ValueError("No reference model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                raise ValueError("No reference model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
        
        #########
        ### setup loss
        #########
        if self.args.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError("Liger is required to use `liger_loss` as the AIXue loss. Run `pip install liger-kernel`.")
            self.liger_aixue_loss = LigerFusedLinearAIXueLoss(
                beta=0.0,
                epsilon_low=self.args.cliprange,
                epsilon_high=self.args.cliprange,
                temperature=self.args.temperature + 1e-7,
            )

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses = data["response_ids"].to(device)
                query_responses = torch.cat((queries, responses), 1)
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = data["reward"].to(device)
                sequence_lengths = []
                for i in tqdm(range(0, queries.shape[0], args.local_rollout_forward_batch_size), desc="rollout rank["+str(accelerator.process_index)+"]"):
                    response = responses[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    output = forward(model, query_response, processing_class.pad_token_id)
                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.temperature + 1e-7
                    logprob = selective_log_softmax(logits, response) # logprob [l_r_f_B, R]
                    ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1 # stop_token_id的位置，不包括stop_token_id的长度
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                postprocessed_responses = torch.cat(postprocessed_responses, 0) # [B, R]
                logprobs = torch.cat(logprobs, 0) # [B, R]
                ref_logprobs = torch.cat(ref_logprobs, 0) # [B, R]
                sequence_lengths = torch.cat(sequence_lengths, 0) # [B]
                del (logprob, ref_logprob)
                torch.cuda.empty_cache()
                gc.collect() # 强制回收内存

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1) # [B, R]
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1) # [B, R]
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB) # [B, R] 不包括eos的logprob
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB) # [B, R]
                sequence_lengths_p1 = sequence_lengths + 1 # [B]
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1)) # [B, R]

                # 4. compute rewards
                # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
                logr = ref_logprobs - logprobs
                kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores # [B, R], 结束位置的reward加上score

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    next_advantage = advantages_reversed[-1] if t < gen_length - 1 else 0.0
                    advantages_reversed.append(rewards[:, t] + args.gamma * next_advantage)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages
                if args.whiten_advantages:
                    advantages = masked_whiten(advantages, ~padding_mask) # 白化advantages，arg控制
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in tqdm(range(args.num_ppo_epochs), desc="ppo_epoch rank["+str(accelerator.process_index)+"]"):
                b_inds = np.random.permutation(args.local_batch_size)
                gradient_accumulation_idx = 0
                for micro_batch_start in tqdm(range(0, args.local_batch_size, args.per_device_train_batch_size), desc="gradient_accumulation rank["+str(accelerator.process_index)+"]"):
                    with accelerator.accumulate(model):
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds, :]
                        mb_return = returns[micro_batch_inds]

                        output = forward(model, mb_query_responses, processing_class.pad_token_id)
                        if self.args.use_liger_loss:
                            last_hidden_state = output.hidden_states[-1][:, context_length - 1 : -1, :]
                            unwrapped_model = self.accelerator.unwrap_model(model)
                            if self.is_deepspeed_enabled:
                                with deepspeed.zero.GatheredParameters(unwrapped_model.lm_head.weight):
                                    loss, temp_metrics = self.liger_aixue_loss(
                                        _input=last_hidden_state,
                                        lin_weight=unwrapped_model.lm_head.weight,
                                        selected_token_ids=mb_responses,
                                        attention_mask=~padding_mask[micro_batch_inds,:],
                                        advantages=mb_advantage,
                                        bias=unwrapped_model.lm_head.bias,
                                        ref_per_token_logps=None,
                                        old_per_token_logps=mb_logprobs,
                                    )
                                    accelerator.backward(loss)
                            else:
                                loss, temp_metrics = self.liger_aixue_loss(
                                    _input=last_hidden_state,
                                    lin_weight=unwrapped_model.lm_head.weight,
                                    selected_token_ids=mb_responses,
                                    attention_mask=~padding_mask[micro_batch_inds,:],
                                    advantages=mb_advantage,
                                    bias=unwrapped_model.lm_head.bias,
                                    ref_per_token_logps=None,
                                    old_per_token_logps=mb_logprobs,
                                )
                                accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()                         
                            with torch.no_grad():
                                pg_clipfrac = temp_metrics[0]
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1) # micro_batch token distribution
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1) # micro_batch entropy
                                approxkl = temp_metrics[3]
                                approxkl_stats[ppo_epoch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, gradient_accumulation_idx] = temp_metrics[1]
                                entropy_stats[ppo_epoch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio = temp_metrics[2]
                                ratio_stats[ppo_epoch_idx, gradient_accumulation_idx] = ratio.mean()
                        else:
                            logits = output.logits[:, context_length - 1 : -1] # [mirco_B, R, V]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses) # [mirco_B, R]
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss 
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1) # micro_batch token distribution
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1) # micro_batch entropy
                                approxkl = 0.5 * (logprobs_diff**2).mean() # micro_batch KL MSE
                                approxkl_stats[ppo_epoch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, gradient_accumulation_idx] = ratio.mean()
                            del (logprobs_diff, logits, new_logprobs, pg_losses, pg_losses2, pg_loss_max, pg_loss)
                    gradient_accumulation_idx += 1
                        
                # del everything and empty cache
                # fmt: off
                del (
                    output, ratio, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                    mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                )
                # fmt: on
                torch.cuda.empty_cache()
            with torch.no_grad():
                print(f"approxkl_stats: {approxkl_stats}")
                print(f"ratio_stats: {ratio_stats}")
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()
            
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            