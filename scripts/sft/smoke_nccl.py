from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor([local_rank + 1], device=f"cuda:{local_rank}", dtype=torch.float32)
    dist.all_reduce(value)
    gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    torch.cuda.synchronize()
    print(
        f"rank={dist.get_rank()} device={torch.cuda.current_device()} "
        f"all_reduce={value.item()} all_gather={[item.item() for item in gathered]}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
