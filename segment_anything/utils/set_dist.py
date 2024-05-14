import torch 

def init_distributed_training(rank, opts):
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:' + str(opts.port),
        world_size=opts.ngpus_per_node,
        rank=opts.rank
    )

    torch.distributed.barrier()

    setup_for_distributed(opts.rank == 0)
    print('opts :', opts)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print