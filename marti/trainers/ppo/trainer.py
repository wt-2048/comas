import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import socket
import deepspeed
import ray

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import math

from marti.models.actor import Actor
from marti.models.loss import GPTLMLoss, PolicyLoss, ValueLoss
from marti.models.model_utils import masked_mean
from marti.helpers.distributed.distributed_sampler import DistributedSampler
from marti.helpers.distributed.distributed_utils import init_process_group

from marti.trainers.experience_maker import Experience, ExperienceMaker
from marti.trainers.kl_controller import AdaptiveKLController, FixedKLController
from marti.trainers.replay_buffer import NaiveReplayBuffer


class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        pretrain_dataloader=None,
        rolename=None,
        eos_token_id: int = -1,
        **generate_kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        assert rolename is not None, "rolename should not be None"
        self.rolename = rolename
        self.eos_token_id = eos_token_id

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)
        self.pretrain_dataloader = pretrain_dataloader

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = ExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            strict_reward=True,
        )

        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )

    # ---------- 统一兜底 old_action_log_probs / base_action_log_probs ----------
    def _ensure_old_logprobs(
        self,
        experience: Experience,
        device: torch.device,
        packed: bool = False,
    ):
        """
        统一处理 old_action_log_probs / base_action_log_probs:
        - 如果本来就有，就搬到 device 上
        - 如果没有，就用 0 张量兜底（相当于 ratio=1，退化成 on-policy 更新）
        """
        old_action_log_probs = getattr(experience, "action_log_probs", None)
        base_action_log_probs = getattr(experience, "base_action_log_probs", None)

        if packed:
            # packed 模式下是 list，每个都是 1D 张量
            if old_action_log_probs is None:
                lengths = [v.numel() for v in experience.advantages]
                old_action_log_probs = [
                    torch.zeros(l, dtype=torch.float32, device=device) for l in lengths
                ]
            else:
                old_action_log_probs = [x.to(device) for x in old_action_log_probs]

            if base_action_log_probs is not None:
                base_action_log_probs = [x.to(device) for x in base_action_log_probs]
        else:
            # 非 packed：action_log_probs 应该是 (1, T) 或 (B, T)
            if old_action_log_probs is None:
                shape = experience.advantages.shape
                old_action_log_probs = torch.zeros(shape, dtype=torch.float32, device=device)
            else:
                old_action_log_probs = old_action_log_probs.to(device)

            if base_action_log_probs is not None:
                base_action_log_probs = base_action_log_probs.to(device)

        return old_action_log_probs, base_action_log_probs

    # ---------- PPO 训练主循环 ----------
    def ppo_train(self, global_steps=0):
        # 若回放缓冲区为空，直接返回空状态
        if len(self.replay_buffer) == 0:
            return {}

        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}] ({self.rolename})",
                disable=not self.strategy.is_rank_0(),
            )
            any_step = False
            for experience in pbar:
                any_step = True
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # KL 用 weighted mean
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status.get("kl", 0.0),
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

            # 这一 epoch 没有任何 batch，跳过
            if not any_step:
                continue

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        # 子类（actor/critic 混合 trainer）会重载这个方法
        pass

    # ---------- 每个 step 外层逻辑 ----------
    def fit(self, steps, samples_ref):
        all_samples = samples_ref
        args = self.args

        # 1) 生成 experience 列表（SFT 模式下跳过 RL）
        experiences: List[Experience] = []
        if args.training_mode != "sft":
            gen_iter = self.experience_maker.make_experience_list(
                all_samples, **self.generate_kwargs
            )
            experiences = list(gen_iter)

        # 没有任何样本，直接返回
        if args.training_mode != "sft" and len(experiences) == 0:
            torch.cuda.empty_cache()
            return {
                "status": {},
                "is_rank_0": self.strategy.is_rank_0(),
                "perf_stats": self.experience_maker.perf_stats,
            }

        # 2) 写入 replay buffer
        for exp in experiences:
            self.replay_buffer.append(exp)

        torch.cuda.empty_cache()

        # 3) 归一化 advantage
        if args.training_mode != "sft" and len(self.replay_buffer) > 0:
            self.replay_buffer.normalize("advantages", self.strategy)

        # 4) PPO 训练
        status = self.ppo_train(steps)

        if args.training_mode != "sft":
            self.replay_buffer.clear()
        torch.cuda.empty_cache()

        if "kl" in status:
            self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

        # 5) 日志 & ckpt
        client_states = {"consumed_samples": steps * args.rollout_batch_size}
        self.save_checkpoints(args, steps, client_states)

        return {
            "status": status,
            "is_rank_0": self.strategy.is_rank_0(),
            "perf_stats": self.experience_maker.perf_stats,
        }

    # ---------- Actor 更新 ----------
    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        device = torch.cuda.current_device()

        # -------- 1) 拆 packed / 非 packed --------
        if isinstance(experience.sequences, list):
            # packed：sequences / advantages 都是 list
            seq_list = [s.to(device).long() for s in experience.sequences]
            sequences = torch.cat(seq_list, dim=0).unsqueeze(0)  # (1, ΣT)

            advantages = torch.cat(
                [v.to(device) for v in experience.advantages], dim=0
            ).unsqueeze(0)

            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]

            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(seq_list)], dim=0
            ).unsqueeze(0).to(device)

            old_action_log_probs, base_action_log_probs = self._ensure_old_logprobs(
                experience, device, packed=True
            )
        else:
            # 非 packed
            sequences = experience.sequences.to(device).long()
            advantages = experience.advantages.to(device)
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask.to(device)

            old_action_log_probs, base_action_log_probs = self._ensure_old_logprobs(
                experience, device, packed=False
            )

        # -------- 2) 前向：只传 input_ids，避免 input_ids + inputs_embeds 冲突 --------
        action_log_probs, output = self.actor(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        # -------- 3) 动作级 mask：截断 + 自身 action_mask --------
        if self.args.mask_truncated_completions:
            if self.eos_token_id is None:
                raise ValueError(
                    "eos_token_id must be set when mask_truncated_completions=True"
                )
            action_mask = _build_truncation_mask(
                sequences,
                packed_seq_lens if packed_seq_lens is not None else [num_actions],
                num_actions,
                self.eos_token_id,
                dtype=torch.bool,
            ).to(device)
        else:
            action_mask = experience.action_mask.to(device).to(dtype=torch.bool)

        # -------- 4) 对齐长度，防止维度不一致 --------
        if isinstance(num_actions, list):
            T = min(
                action_log_probs.size(-1),
                old_action_log_probs.size(-1),
                advantages.size(-1),
                action_mask.size(-1),
            )
        else:
            T = min(
                action_log_probs.size(-1),
                old_action_log_probs.size(-1),
                advantages.size(-1),
                action_mask.size(-1),
            )

        action_log_probs = action_log_probs[..., :T]
        old_action_log_probs = old_action_log_probs[..., :T]
        advantages = advantages[..., :T]
        action_mask = action_mask[..., :T]

        # -------- 5) Policy loss --------
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=action_mask,
        )

        # -------- 6) KL 项（可选） --------
        if self.args.use_kl_loss and (base_action_log_probs is not None):
            base_action_log_probs = base_action_log_probs[..., :T]
            kl_term = action_log_probs - base_action_log_probs
            if getattr(self.args, "use_kl_estimator_k3", False):
                r = (-kl_term).exp()
                kl = r - 1.0 + kl_term
            else:
                kl = kl_term
            kl = masked_mean(kl, action_mask, dim=-1).mean()
        else:
            kl = torch.tensor(0.0, device=device)

        # -------- 7) 额外 loss（Mixtral aux） --------
        aux_loss = getattr(output, "aux_loss", 0.0) if self.aux_loss else 0.0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl * self.kl_ctl.value

        self.strategy.backward(loss, self.actor, self.actor_optim)

        # -------- 8) （可选）PTX 监督损失 --------
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(device)
            attn = data[2].squeeze(1).to(device)
            label = torch.where(attn.bool(), inputs, self.ptx_loss_fn.IGNORE_INDEX)
            out = self.actor(inputs, attention_mask=attn, return_output=True)
            ptx_logits = out["logits"]
            ptx_loss = self.ptx_loss_fn(ptx_logits, label)
            if self.aux_loss:
                aux2 = out.aux_loss
            else:
                aux2 = 0.0
            loss2 = ptx_loss + aux2 * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss2, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # -------- 9) 统计 --------
        status: Dict[str, float] = {
            "policy_loss": float(actor_loss.item()),
            "actor_lr": float(self.actor_scheduler.get_last_lr()[0]),
        }
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = float(ptx_loss.item())
        if self.args.use_kl_loss:
            status["kl"] = float(kl.detach().item())
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum()
                    / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status

    # ---------- Critic 更新 ----------
    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()
        device = torch.cuda.current_device()

        if isinstance(experience.sequences, list):
            sequences = torch.cat(
                [s.to(device) for s in experience.sequences], dim=0
            ).unsqueeze(0)
            old_values = torch.cat(
                [v.to(device) for v in experience.values], dim=0
            ).unsqueeze(0)
            returns = torch.cat(
                [r.to(device) for r in experience.returns], dim=0
            ).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)],
                dim=0,
            ).unsqueeze(0).to(device)
        else:
            sequences = experience.sequences.to(device)
            old_values = experience.values.to(device)
            returns = experience.returns.to(device)
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask.to(device)

        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask.to(device),
        )

        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0.0

        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(
            self.critic_optim, self.critic, self.critic_scheduler, name="critic"
        )

        status = {
            "critic_loss": float(critic_loss.item()),
            "values": float(masked_mean(values, experience.action_mask.to(device)).item()),
            "critic_lr": float(self.critic_scheduler.get_last_lr()[0]),
        }
        return status

    # ---------- 保存 ckpt ----------
    def save_checkpoints(self, args, global_step, client_states={}):
        if global_step % args.save_steps == 0:
            return

    def _save_checkpoint(self, args, tag, client_states):
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_" + self.rolename),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if self.critic is not None:
            self.strategy.save_ckpt(
                self.critic,
                os.path.join(args.ckpt_path, "_" + self.critic.rolename),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
            )


# ---------- EOS 截断用的动作掩码构造 ----------
def _build_truncation_mask(
    sequences: torch.Tensor,                       # (1, Σ packed_seq_lens)
    packed_seq_lens: Union[List[int], torch.Tensor],
    num_actions: Union[List[int], torch.Tensor, int],
    eos_token_id: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if eos_token_id is None:
        raise ValueError("eos_token_id must be set when building truncation mask")

    device = sequences.device

    # ---- 转成 tensor ----
    if not isinstance(packed_seq_lens, torch.Tensor):
        packed_seq_lens = torch.as_tensor(packed_seq_lens, dtype=torch.long, device=device)
    else:
        packed_seq_lens = packed_seq_lens.to(device=device, dtype=torch.long)

    if not isinstance(num_actions, torch.Tensor):
        num_actions = torch.as_tensor(num_actions, dtype=torch.long, device=device)
    else:
        num_actions = num_actions.to(device=device, dtype=torch.long)

    # ---- 关键修复点：非 packed 情况下 num_actions 是标量，展开成和 packed_seq_lens 一样的 shape ----
    if num_actions.dim() == 0:
        num_actions = num_actions.expand_as(packed_seq_lens)

    assert packed_seq_lens.shape == num_actions.shape, \
        "`packed_seq_lens` and `num_actions` must have the same length (batch size)"
    assert sequences.numel() == packed_seq_lens.sum().item(), \
        "Sum(packed_seq_lens) must equal len(flattened sequences)"

    # ---- 展平序列并取每个样本的最后一个 token ----
    seq_flat = sequences.view(-1)                           # (Σ packed_seq_lens,)
    last_token_indices = packed_seq_lens.cumsum(0) - 1      # (B,)
    ends_with_eos = seq_flat[last_token_indices] == eos_token_id  # bool (B,)

    # ---- 按样本构造 mask，再 repeat_interleave 到动作维度 ----
    sample_mask = ends_with_eos.to(dtype=dtype)             # (B,)
    action_mask = sample_mask.repeat_interleave(num_actions)  # (Σ num_actions,)

    return action_mask.unsqueeze(0)                         # (1, Σ num_actions)

