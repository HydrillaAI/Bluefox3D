import os
import io
from datetime import timedelta
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_dist(rank, local_rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(hours=10))
    

def read_file_dist(path):
    """
    Read a checkpoint file.

    Each rank reads from local disk. Broadcasting a multi-GB file as a CUDA
    ByteTensor OOMs 24 GB L4s (model already occupies most of VRAM).
    """
    with open(path, 'rb') as f:
        data = f.read()
    return io.BytesIO(data)
    

def unwrap_dist(model):
    """
    Unwrap the model from distributed training.
    """
    if isinstance(model, DDP):
        return model.module
    return model


@contextmanager
def master_first():
    """
    A context manager that ensures master process executes first.
    """
    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
            

@contextmanager
def local_master_first():
    """
    A context manager that ensures local master process executes first.
    """
    if not dist.is_initialized():
        yield
    else:
        if dist.get_rank() % torch.cuda.device_count() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
    