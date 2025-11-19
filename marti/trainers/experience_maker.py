# -*- coding: utf-8 -*-
from __future__ import annotations
import logging, re, math
from dataclasses import dataclass
from typing import Any, List, Sequence, Union, Optional, Tuple, Dict

import torch
from torch import Tensor

LOG = logging.getLogger(__name__)
_once_flags = {"warned_empty": False}

_NUM = r"-?\d+(?:\.\d+)?"

def _first_number(text: str):
    m = re.search(_NUM, text or "")
    return float(m.group(0)) if m else None

def _safe_score_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"<\s*score\s*>\s*([0-9]+(?:\.[0-9]+)?)\s*<\s*/\s*score\s*>", text, re.I | re.S)
    if m:
        return float(m.group(1))
    return None

def _fallback_gt_reward(task: str, pred_text: str, gold_text: str) -> float:
    """没有显式 <score> 时的回退打分（非常粗糙，只为避免空奖励把训练卡死）"""
    if not (pred_text and gold_text):
        return 0.0
    task = (task or "").lower()
    if any(k in task for k in ("math", "science", "chem", "phys")):
        pv, gv = _first_number(pred_text), _first_number(gold_text)
        if pv is None or gv is None:
            return 0.0
        ok = math.isclose(pv, gv, rel_tol=1e-2, abs_tol=1e-2)
        return 10.0 if ok else 0.0
    if "coding" in task:
        pt = pred_text.strip()
        gt = gold_text.strip()
        return 10.0 if (pt == gt or gt in pt) else 0.0
    return 10.0 if pred_text.strip() == gold_text.strip() else 0.0


@dataclass
class Experience:
    # ---- PPO/RL 必要字段 ----
    rewards: Optional[Tensor] = None      # (1,T) 或 (T,)
    values: Optional[Tensor] = None       # (1,T) 或 (T,)
    dones: Optional[Tensor] = None        # (1,T) 或 (T,)
    action_mask: Optional[Tensor] = None  # (1,T) 或 (T,)
    advantages: Optional[Tensor] = None   # (1,T) 或 (T,)
    returns: Optional[Tensor] = None      # (1,T) 或 (T,)

    # ---- LM 前向/对齐字段（训练 step 必需）----
    sequences: Optional[Tensor] = None          # (1,T) long -> input_ids
    attention_mask: Optional[Tensor] = None     # (1,T) float/bool
    position_ids: Optional[Tensor] = None       # (1,T) long
    action_log_probs: Optional[Tensor] = None
    base_action_log_probs: Optional[Tensor] = None

    info: Optional[Dict[str, Any]] = None

    def to_device(self, device: torch.device) -> "Experience":
        def _mv(x):
            return x.to(device) if isinstance(x, Tensor) else x
        for name in (
            "rewards","values","dones","action_mask","advantages","returns",
            "sequences","attention_mask","position_ids",
            "action_log_probs","base_action_log_probs",
        ):
            v = getattr(self, name, None)
            if isinstance(v, Tensor):
                setattr(self, name, _mv(v))
        return self

    def to_cpu(self) -> "Experience":
        return self.to_device(torch.device("cpu"))


class ExperienceMaker:
    """
    strict_reward=True：若样本无法得到 reward（显式或可推断）则丢弃；
    strict_reward=False：若 reward 缺失，用 0 兜底并告警一次（避免训练直接中断）。
    """
    def __init__(self, *args, strict_reward: bool = False, **kwargs) -> None:
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.gae_lambda = float(kwargs.get("gae_lambda", kwargs.get("lambd", 0.95)))
        device = kwargs.get("device")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.strict_reward = bool(strict_reward)

        self.experiences: List[Experience] = []
        self.perf_stats: Dict[str, float] = {
            "num_samples": 0.0,
            "num_built": 0.0,
            "num_skipped_empty": 0.0,
            "total_steps": 0.0,
            "mean_steps": 0.0,
        }

    def flush(self):
        return

    # ---------------- helpers ----------------
    @staticmethod
    def _to_float_list(x: Any) -> List[float]:
        if x is None:
            return []
        if isinstance(x, (int, float)):
            return [float(x)]
        if isinstance(x, Tensor):
            return x.detach().cpu().flatten().float().tolist()
        if isinstance(x, (list, tuple)):
            out: List[float] = []
            for v in x:
                out.extend(ExperienceMaker._to_float_list(v))
            return out
        try:
            return [float(x)]
        except Exception:
            return []

    @staticmethod
    def _get_field(sample: Any, key: str, default=None) -> Any:
        if sample is None:
            return default
        cur = sample
        for part in key.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                cur = getattr(cur, part, default)
            if cur is default:
                break
        return cur

    @staticmethod
    def _find_score_anywhere(obj: Any) -> Optional[float]:
        """递归扫描任意结构，字符串中出现 <score>…</score> 则取出。"""
        try:
            if isinstance(obj, str):
                sc = _safe_score_from_text(obj)
                return float(sc) if sc is not None else None
            if isinstance(obj, dict):
                for v in obj.values():
                    sc = ExperienceMaker._find_score_anywhere(v)
                    if sc is not None:
                        return sc
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    sc = ExperienceMaker._find_score_anywhere(v)
                    if sc is not None:
                        return sc
            if hasattr(obj, "__dict__"):
                return ExperienceMaker._find_score_anywhere(vars(obj))
        except Exception:
            pass
        return None

    def _safe_tensor_1d(self, x: Any, name: str, dtype=torch.float32, device=None) -> Tensor:
        if device is None:
            device = self.device
        if x is None:
            return torch.tensor([], dtype=dtype, device=device)
        if torch.is_tensor(x):
            t = x.detach().to(device=device)
            return t.flatten().to(dtype=dtype)
        lst = self._to_float_list(x)
        if len(lst) == 0:
            return torch.tensor([], dtype=dtype, device=device)
        return torch.tensor(lst, dtype=dtype, device=device).flatten()

    def _safe_long_1d(self, x: Any, device=None) -> Tensor:
        if device is None:
            device = self.device
        if x is None:
            return torch.tensor([], dtype=torch.long, device=device)
        if torch.is_tensor(x):
            return x.detach().to(device=device, dtype=torch.long).flatten()
        try:
            if isinstance(x, (list, tuple)):
                return torch.tensor(list(x), dtype=torch.long, device=device).flatten()
        except Exception:
            pass
        return torch.tensor([], dtype=torch.long, device=device)

    @staticmethod
    def _pad_or_trunc(x: Tensor, T: int, name: str, fill: float = 0.0) -> Tensor:
        x = x.flatten()
        n = x.numel()
        if n == T:
            return x
        if n == 0:
            return torch.full((T,), float(fill), dtype=x.dtype, device=x.device)
        if n < T:
            pad = torch.full((T - n,), float(fill), dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=0)
        return x[:T]

    @staticmethod
    def _first_nonzero_len(*tensors: Tensor) -> int:
        for t in tensors:
            if t is not None and torch.is_tensor(t) and t.numel() > 0:
                return t.numel()
        return 1

    def _infer_reward_from_sample(self, s: Any) -> Optional[float]:
        # 1) 显式字段
        for k in ("score", "reward", "rewards", "scores"):
            v = self._get_field(s, k, None)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                return float(v[0])
        # 2) 常见文本字段
        for k in ("evaluation", "evaluation_text", "evaluator_output", "critic", "judge",
                  "grader", "review", "feedback", "score_text", "evaluation_result"):
            txt = self._get_field(s, k, None)
            if isinstance(txt, str):
                sc = _safe_score_from_text(txt)
                if sc is not None:
                    return float(sc)
        # 3) 递归兜底
        sc_any = self._find_score_anywhere(s)
        if sc_any is not None:
            return float(sc_any)
        # 4) 再兜底：与标准答案对比
        task = self._get_field(s, "task_name", "") or self._get_field(s, "task", "")
        pred = (self._get_field(s, "prediction", None)
                or self._get_field(s, "solution", None)
                or self._get_field(s, "answer", None)
                or self._get_field(s, "output", None))
        gold = (self._get_field(s, "gold", None)
                or self._get_field(s, "reference", None)
                or self._get_field(s, "target", None)
                or self._get_field(s, "label", None)
                or self._get_field(s, "gt", None))
        pred_text = pred if isinstance(pred, str) else (self._get_field(pred, "text", None) or self._get_field(pred, "output", None))
        gold_text = gold if isinstance(gold, str) else (self._get_field(gold, "text", None) or self._get_field(gold, "output", None))
        if pred_text is not None or gold_text is not None:
            return _fallback_gt_reward(task, pred_text or "", gold_text or "")
        return None

    # ---------------- GAE ----------------
    def get_advantages_and_returns(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        action_mask: Optional[Tensor] = None,
        gamma: Optional[float] = None,
        gae_lambda: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        gamma = self.gamma if gamma is None else float(gamma)
        gae_lambda = self.gae_lambda if gae_lambda is None else float(gae_lambda)
        device = self.device if device is None else device

        r = rewards.flatten().to(device=device, dtype=torch.float32)
        v = values.flatten().to(device=device, dtype=torch.float32)
        d = dones .flatten().to(device=device, dtype=torch.float32)

        T = r.numel()
        if T == 0:
            z = torch.zeros(0, device=device, dtype=torch.float32)
            return z, z

        last_v = (v[-1].item() if v.numel() > 0 else 0.0)
        v = self._pad_or_trunc(v, T, "values", fill=last_v)
        d = self._pad_or_trunc(d, T, "dones",  fill=0.0)

        if action_mask is None or action_mask.numel() == 0:
            am = torch.ones(T, dtype=torch.float32, device=device)
        else:
            am = action_mask.flatten().to(device=device, dtype=torch.float32)
            am = self._pad_or_trunc(am, T, "action_mask", fill=1.0)

        rewards = r.view(1, -1) * am.view(1, -1)
        values  = v.view(1, -1) * am.view(1, -1)
        dones   = d.view(1, -1)
        next_values = torch.cat([values[..., 1:], torch.zeros_like(values[..., :1])], dim=-1)

        advantages = torch.zeros_like(rewards, dtype=values.dtype, device=device)
        last_gae   = torch.zeros_like(rewards[..., 0], dtype=values.dtype, device=device)

        for t in reversed(range(rewards.shape[-1])):
            nonterminal = (1.0 - dones[..., t].float())
            delta = rewards[..., t] + gamma * next_values[..., t] * nonterminal - values[..., t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[..., t] = last_gae

        returns = advantages + values
        advantages = advantages * am.view(1, -1)
        returns    = returns    * am.view(1, -1)
        return advantages, returns

    # ---------------- samples → Experience 列表 ----------------
    def make_experience_list(self, all_samples: Union[Any, Sequence[Any]], **kwargs) -> List[Experience]:
        self.experiences = []
        self.perf_stats.update({k: 0.0 for k in self.perf_stats})

        if isinstance(all_samples, (dict,)) or hasattr(all_samples, "__dict__"):
            samples = [all_samples]
        elif isinstance(all_samples, (list, tuple)):
            samples = list(all_samples)
        else:
            samples = [all_samples]

        self.perf_stats["num_samples"] = float(len(samples))
        tmp: List[dict] = []

        for s in samples:
            # 取 RL 量
            rewards = self._safe_tensor_1d(self._get_field(s, "rewards"), "rewards", device=self.device)
            values  = self._safe_tensor_1d(self._get_field(s, "values"),  "values",  device=self.device)
            dones   = self._safe_tensor_1d(self._get_field(s, "dones"),   "dones",   device=self.device)
            action_mask = self._safe_tensor_1d(self._get_field(s, "action_mask"), "action_mask", device=self.device)

            # 取 LM 量（允许缺失，后面兜底）
            pad_id = int(self._get_field(s, "pad_token_id", 0) or 0)
            seq_1d = self._safe_long_1d(self._get_field(s, "sequences"))
            att_1d = self._safe_tensor_1d(self._get_field(s, "attention_mask"), "attention_mask", dtype=torch.float32)
            pos_1d = self._safe_long_1d(self._get_field(s, "position_ids"))

            # 若缺奖励，尽量解析 <score>，再不行回退
            if rewards.numel() == 0:
                sc = self._infer_reward_from_sample(s)
                if sc is not None:
                    T_guess = self._first_nonzero_len(values, dones, action_mask, att_1d, seq_1d)
                    T_guess = max(T_guess, 1)
                    rewards = torch.full((T_guess,), float(sc), dtype=torch.float32, device=self.device)

            if self.strict_reward and rewards.numel() == 0:
                self.perf_stats["num_skipped_empty"] += 1.0
                if not _once_flags["warned_empty"]:
                    LOG.warning("Sample dropped due to missing reward. "
                                "请确保 evaluator 输出 <score>… 并写入样本/rewards。")
                    _once_flags["warned_empty"] = True
                continue

            if rewards.numel() == 0:
                if not _once_flags["warned_empty"]:
                    LOG.warning("Empty rewards detected; fill zeros once (训练会继续，但信号较弱)。")
                    _once_flags["warned_empty"] = True
                T_guess = self._first_nonzero_len(values, dones, action_mask, att_1d, seq_1d)
                T_guess = max(T_guess, 1)
                rewards = torch.zeros(T_guess, dtype=torch.float32, device=self.device)

            # 对齐长度 T
            T = max(1, int(rewards.numel()))
            last_v = values[-1].item() if values.numel() > 0 else 0.0
            values = self._pad_or_trunc(values, T, "values", fill=last_v)
            dones  = self._pad_or_trunc(dones,  T, "dones",  fill=0.0)
            if action_mask.numel() == 0:
                action_mask = torch.ones(T, dtype=torch.float32, device=self.device)
            else:
                action_mask = self._pad_or_trunc(action_mask, T, "action_mask", fill=1.0).clamp_(0.0, 1.0)

            # LM 对齐：若样本没带，就兜底出一个与 T 等长的序列与 mask
            if seq_1d.numel() == 0:
                seq_1d = torch.full((T,), pad_id, dtype=torch.long, device=self.device)
            else:
                seq_1d = self._pad_or_trunc(seq_1d, T, "sequences", fill=pad_id).to(dtype=torch.long)

            if att_1d.numel() == 0:
                att_1d = torch.ones(T, dtype=torch.float32, device=self.device)
            else:
                att_1d = self._pad_or_trunc(att_1d, T, "attention_mask", fill=1.0).clamp_(0.0, 1.0)

            if pos_1d.numel() == 0:
                pos_1d = torch.arange(T, dtype=torch.long, device=self.device)
            else:
                pos_1d = self._pad_or_trunc(pos_1d, T, "position_ids", fill=0).to(dtype=torch.long)

            # 计算 GAE
            adv, ret = self.get_advantages_and_returns(
                rewards=rewards, values=values, dones=dones, action_mask=action_mask,
                gamma=kwargs.get("gamma", None), gae_lambda=kwargs.get("gae_lambda", None),
                device=kwargs.get("device", None),
            )
            if adv.numel() == 0:
                self.perf_stats["num_skipped_empty"] += 1.0
                continue

            tmp.append(dict(
                rewards=rewards.flatten(), values=values.flatten(), dones=dones.flatten(),
                action_mask=action_mask.flatten(),
                advantages=adv.flatten(), returns=ret.flatten(),
                sequences=seq_1d.flatten().to(dtype=torch.long),
                attention_mask=att_1d.flatten(),
                position_ids=pos_1d.flatten().to(dtype=torch.long),
            ))

        if len(tmp) == 0:
            self.perf_stats["mean_steps"] = 0.0
            return self.experiences

        # 统一到相同的 max_T
        max_T = max(int(t["advantages"].numel()) for t in tmp)
        total_steps = 0.0
        for t in tmp:
            last_v = t["values"][-1].item() if t["values"].numel() > 0 else 0.0
            r  = self._pad_or_trunc(t["rewards"],        max_T, "rewards",        fill=0.0).view(1, -1)
            v  = self._pad_or_trunc(t["values"],         max_T, "values",         fill=last_v).view(1, -1)
            d  = self._pad_or_trunc(t["dones"],          max_T, "dones",          fill=0.0).view(1, -1)
            am = self._pad_or_trunc(t["action_mask"],    max_T, "action_mask",    fill=1.0).clamp_(0.0, 1.0).view(1, -1)
            ad = self._pad_or_trunc(t["advantages"],     max_T, "advantages",     fill=0.0).view(1, -1)
            rt = self._pad_or_trunc(t["returns"],        max_T, "returns",        fill=0.0).view(1, -1)
            seq= self._pad_or_trunc(t["sequences"],      max_T, "sequences",      fill=0).to(dtype=torch.long).view(1, -1)
            att= self._pad_or_trunc(t["attention_mask"], max_T, "attention_mask", fill=1.0).clamp_(0.0, 1.0).view(1, -1)
            pos= self._pad_or_trunc(t["position_ids"],   max_T, "position_ids",   fill=0).to(dtype=torch.long).view(1, -1)

            exp = Experience(
                rewards=r.to(self.device), values=v.to(self.device), dones=d.to(self.device),
                action_mask=am.to(self.device), advantages=ad.to(self.device), returns=rt.to(self.device),
                sequences=seq.to(self.device), attention_mask=att.to(self.device), position_ids=pos.to(self.device),
                action_log_probs=None, base_action_log_probs=None, info={},
            )
            self.experiences.append(exp)
            total_steps += float(ad.numel())

        self.perf_stats["num_built"] = float(len(self.experiences))
        self.perf_stats["total_steps"] = total_steps
        self.perf_stats["mean_steps"] = (total_steps / self.perf_stats["num_built"]) if self.perf_stats["num_built"] > 0 else 0.0
        return self.experiences