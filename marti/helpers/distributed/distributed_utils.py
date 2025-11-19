import os
import torch
import torch.distributed as dist

__all__ = [
    "init_process_group",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "torch_dist_barrier_and_cuda_sync",
]

def _dist_available():
    return dist.is_available()

def _dist_initialized():
    return _dist_available() and dist.is_initialized()

def init_process_group(*args, **kwargs):
    """
    单机单卡训练：不主动初始化进程组，交给 DeepSpeed/Trainer 需要时再处理。
    保持函数存在以满足 import。
    """
    return False

def get_rank():
    if _dist_initialized():
        try:
            return dist.get_rank()
        except Exception:
            return 0
    return 0

def get_world_size():
    if _dist_initialized():
        try:
            return dist.get_world_size()
        except Exception:
            return 1
    return 1

def is_main_process():
    return get_rank() == 0

def barrier():
    """
    只在多卡时执行 barrier，单卡直接跳过，避免 NCCL 在单卡环境触发错误。
    """
    if _dist_initialized():
        try:
            if get_world_size() > 1:
                dist.barrier()
        except Exception:
            # 单卡或后端不一致时直接略过
            pass

def torch_dist_barrier_and_cuda_sync(device: int = 0):
    """
    训练代码里的通用同步点：多卡才 barrier；GPU 有就做一次 cuda 同步。
    """
    barrier()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
