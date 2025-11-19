import random
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from marti.trainers.experience_maker import Experience


@dataclass
class BufferItem:
    """
    统一的单条样本容器；字段允许为 None（例如无 LM token 时 sequences 可为 None）
    shapes:
      sequences: (S)
      action_log_probs/base_action_log_probs: (A) 或 None
      values/returns/advantages: (A) 或 (T)
      attention_mask: (S) 或 None
      action_mask: (A) 或 (T) 或 None
    """
    sequences: Optional[torch.Tensor]
    action_log_probs: Optional[torch.Tensor]
    base_action_log_probs: Optional[torch.Tensor]
    values: Optional[torch.Tensor]
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]
    info: Optional[dict]


def zero_pad_sequences(sequences: List[Optional[torch.Tensor]], side: str = "left") -> Optional[torch.Tensor]:
    seqs = [s for s in sequences if s is not None]
    if len(seqs) == 0:
        return None
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in seqs)
    padded = []
    for s in sequences:
        if s is None:
            padded.append(torch.zeros(max_len, dtype=torch.long, device=seqs[0].device))
            continue
        pad_len = max_len - s.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded.append(F.pad(s, padding))
    return torch.stack(padded, dim=0)


def make_experience_batch(items: List[BufferItem], packing_samples: bool = False) -> Experience:
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    kwargs = {}
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            if key in ("sequences", "attention_mask"):
                batch_data = zero_pad_sequences(vals, "left")
            else:
                if vals[0] is None:
                    batch_data = None
                else:
                    # 对齐到同一长度（右侧对齐），用 0 pad
                    max_len = max(v.size(0) for v in vals if v is not None)
                    padded = []
                    for v in vals:
                        if v is None:
                            padded.append(torch.zeros(max_len, device=vals[0].device, dtype=vals[0].dtype))
                        else:
                            if v.size(0) < max_len:
                                pad = torch.zeros(max_len - v.size(0), device=v.device, dtype=v.dtype)
                                padded.append(torch.cat([v, pad], dim=0))
                            else:
                                padded.append(v[:max_len])
                    batch_data = torch.stack(padded, dim=0)
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    # info（标量）批
    infos = [it.info or {} for it in items]
    common_keys = set().union(*[set(i.keys()) for i in infos]) if infos else set()
    info_batch = {}
    for k in common_keys:
        vals = []
        for it in infos:
            v = it.get(k, 0)
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.item()
            vals.append(float(v) if isinstance(v, (int, float)) else 0.0)
        info_batch[k] = torch.tensor(vals, dtype=torch.float32)
    kwargs["info"] = info_batch

    return Experience(**kwargs)


class NaiveReplayBuffer:
    def __init__(
        self,
        sample_batch_size: int = 32,
        limit: int = 0,
        cpu_offload: bool = True,
        packing_samples: bool = False,
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        self.limit = limit  # <=0 即无限
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
        self.items: List[BufferItem] = []

    def _exp_to_buffer_items(self, exp: Experience) -> List[BufferItem]:
        """
        将 Experience（可能没有 sequences）转换为若干 BufferItem。
        - 若 exp.sequences 存在且是 (B, S)，按批拆分；
        - 否则认为 exp 是单条（按 advantages/returns 的长度 T）。
        """
        out: List[BufferItem] = []
        # 批情况（旧范式）
        if isinstance(exp.sequences, torch.Tensor) and exp.sequences.dim() == 2:
            B = exp.sequences.size(0)
            def _split(t):
                if t is None:
                    return [None] * B
                return list(torch.unbind(t, dim=0))
            seqs = _split(exp.sequences)
            alp  = _split(exp.action_log_probs)
            balp = _split(exp.base_action_log_probs)
            vals = _split(exp.values)
            rets = _split(exp.returns)
            advs = _split(exp.advantages)
            amsk = _split(exp.attention_mask)
            acms = _split(exp.action_mask)
            for i in range(B):
                out.append(BufferItem(
                    sequences=seqs[i], action_log_probs=alp[i], base_action_log_probs=balp[i],
                    values=vals[i], returns=rets[i], advantages=advs[i],
                    attention_mask=amsk[i], action_mask=acms[i], info=exp.info or {},
                ))
            return out

        # 单条（无 sequences）：从 RL 信号推断长度
        # 取 T = advantages/returns/values 中非空的长度
        cand = [t for t in (exp.advantages, exp.returns, exp.values) if isinstance(t, torch.Tensor) and t.numel() > 0]
        if len(cand) == 0:
            # 没有有效长度，丢弃
            return out
        T = cand[0].reshape(-1).size(0)

        def _flat_or_none(t):
            if t is None:
                return None
            t = t.reshape(-1)
            return t

        # 用 1 作为 attention_mask，action_mask 若缺省则为 1
        attn = torch.ones(T, dtype=torch.long, device=cand[0].device)
        actm = _flat_or_none(exp.action_mask)
        if actm is None or actm.numel() == 0:
            actm = torch.ones(T, dtype=torch.float32, device=cand[0].device)

        out.append(BufferItem(
            sequences=None,
            action_log_probs=None,
            base_action_log_probs=None,
            values=_flat_or_none(exp.values),
            returns=_flat_or_none(exp.returns),
            advantages=_flat_or_none(exp.advantages),
            attention_mask=attn,
            action_mask=actm,
            info=exp.info or {},
        ))
        return out

    @torch.no_grad()
    def append(self, experience):
        """
        允许传入 Experience 或 BufferItem 或它们的列表。
        内部统一转 BufferItem，并按需落到 CPU。
        """
        items: List[BufferItem] = []
        if isinstance(experience, list):
            for e in experience:
                if isinstance(e, BufferItem):
                    items.append(e)
                elif isinstance(e, Experience):
                    items.extend(self._exp_to_buffer_items(e))
        else:
            if isinstance(experience, BufferItem):
                items.append(experience)
            elif isinstance(experience, Experience):
                items.extend(self._exp_to_buffer_items(experience))

        # 落到 CPU（便于大缓存）
        for it in items:
            for name in (
                "sequences","action_log_probs","base_action_log_probs",
                "values","returns","advantages","attention_mask","action_mask",
            ):
                t = getattr(it, name, None)
                if isinstance(t, torch.Tensor):
                    setattr(it, name, t.to("cpu"))
        self.items.extend(items)

        # 限长
        if self.limit > 0 and len(self.items) > self.limit:
            self.items = self.items[-self.limit:]

    def get_sample_batch_size(self):
        return self.sample_batch_size

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, min(self.sample_batch_size, len(self.items)))
        exp = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload and torch.cuda.is_available():
            exp.to_device(self.target_device)
        return exp

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        exp = make_experience_batch(batch, self.packing_samples)
        if self.cpu_offload and torch.cuda.is_available():
            exp.to_device(self.target_device)
        return exp

    def normalize(self, attribute: str, strategy) -> None:
        """对 advantages 做全局标准化，兼容长度不一致并支持 action_mask。"""
        assert attribute == "advantages"
        flat_items, masks, valid_idx = [], [], []
        for idx in range(len(self.items)):
            adv = getattr(self.items[idx], attribute, None)
            if adv is None:
                continue
            adv = adv.reshape(-1).float()
            if adv.numel() == 0:
                continue
            flat_items.append(adv)
            m = getattr(self.items[idx], "action_mask", None)
            masks.append(None if m is None else m.reshape(-1))
            valid_idx.append(idx)

        if len(flat_items) == 0:
            return

        items_vector = torch.cat(flat_items).to(flat_items[0].device)
        if all(m is None for m in masks):
            action_masks_vector = None
            num_actions_local = float(items_vector.numel())
        else:
            ms = []
            for m, x in zip(masks, flat_items):
                if m is None:
                    ms.append(torch.ones_like(x, dtype=torch.float32, device=items_vector.device))
                else:
                    ms.append(m.to(items_vector.device).float())
            action_masks_vector = torch.cat(ms)
            num_actions_local = float(action_masks_vector.sum().item())

        cnt_tensor = torch.tensor([num_actions_local], device=items_vector.device, dtype=torch.float32)
        total_actions_tensor = strategy.all_reduce(cnt_tensor, "sum")
        total_actions = float(total_actions_tensor[0].item() if total_actions_tensor.ndim > 0 else total_actions_tensor.item())
        if total_actions <= 0.0:
            return

        sum_local = torch.tensor([items_vector.sum().item()], device=items_vector.device, dtype=torch.float32)
        all_sum_tensor = strategy.all_reduce(sum_local, "sum")
        all_sum = float(all_sum_tensor[0].item() if all_sum_tensor.ndim > 0 else all_sum_tensor.item())
        mean = all_sum / max(total_actions, 1.0)

        if action_masks_vector is None:
            masked_sq_local = (items_vector - mean).pow(2).sum()
            denom_local = torch.tensor([items_vector.numel()], device=items_vector.device, dtype=torch.float32)
        else:
            masked_sq_local = ((items_vector - mean).pow(2) * action_masks_vector).sum()
            denom_local = torch.tensor([action_masks_vector.sum().item()], device=items_vector.device, dtype=torch.float32)

        all_sq = strategy.all_reduce(masked_sq_local, "sum")
        all_denom_tensor = strategy.all_reduce(denom_local, "sum")
        all_denom = float(all_denom_tensor[0].item() if all_denom_tensor.ndim > 0 else all_denom_tensor.item())
        rstd = ((all_sq / max(all_denom, 1.0)).clamp(min=1e-8)).rsqrt()

        offset = 0
        for idx, adv in zip(valid_idx, flat_items):
            n = adv.numel()
            seg = items_vector[offset:offset + n]
            offset += n
            setattr(self.items[idx], attribute, (seg - mean) * rstd)