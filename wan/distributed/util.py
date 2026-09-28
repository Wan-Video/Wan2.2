# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


def get_device_type() -> str:
    """Device type of the active accelerator ('cpu' when there is none)."""
    for name in ('cuda', 'npu', 'xpu', 'mps'):
        backend = getattr(torch, name, None)
        if backend is not None and backend.is_available():
            return name
    return 'cpu'


def get_current_device():
    """Current device of the active accelerator as a `torch.device` ('cpu' when there is none)."""
    device_type = get_device_type()
    if device_type == 'cpu':
        return torch.device('cpu')
    return torch.device(device_type, getattr(torch, device_type).current_device())


def set_device(index: int) -> None:
    """Bind the calling process to `index` on the active accelerator."""
    device_type = get_device_type()
    if device_type == 'cpu':
        return
    getattr(torch, device_type).set_device(index)


def get_distributed_backend() -> str:
    """Collective communication backend matching the active accelerator."""
    return {'cuda': 'nccl', 'npu': 'hccl', 'xpu': 'xccl'}.get(get_device_type(), 'gloo')


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=get_distributed_backend())


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()
