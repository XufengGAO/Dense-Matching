import sys
from torch.utils.data.distributed import DistributedSampler
from data.download import load_dataset
from torch.utils.data import DataLoader, Subset
import torch
from .logger import Logger
from torch import distributed as dist
import os
import torch.optim as optim
from .loss import StrongCrossEntropyLoss, StrongFlowLoss, WeakCEMatchLoss

def init_distributed_mode(args):
    """init for distribute mode"""
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.local_rank = args.rank % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        args.rank, args.local_rank, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    args.dist_backend = "nccl"
    args.distributed = True
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.local_rank)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier()

    dist_dict = {'rank':args.rank, 'world_size':args.world_size, 'local_rank':args.local_rank,\
                 'dist_backend':args.dist_backend, 'distributed':args.distributed, 'dist_url':args.dist_url }

    return dist_dict

def build_dataloader(data, datapath, batch_size, rank, world_size):
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else False

    Logger.info("Loading %s %s dataset" % (data.benchmark, data.type))

    dataset = load_dataset(
        data.benchmark,
        datapath,
        data.thres,
        data.split,
        img_size=data.img_size,
    )
    data_sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=data.sampler.shuffle, drop_last=data.sampler.drop_last
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=(data_sampler is None),
        num_workers=num_workers, pin_memory=pin_memory, sampler=data_sampler)

    Logger.info(
        f"Data loaded: there are {len(dataloader.dataset)} {data.type} images"
    )

    # sub-validation set
    aux_loader = None
    if data.split == 'val':
        if len(dataloader.sampler) * world_size < len(dataloader.dataset):
            aux_dataset = Subset(dataloader.dataset, range(len(dataloader.sampler)*world_size, len(dataloader.dataset)))
            aux_loader = DataLoader(aux_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

            Logger.info('Create Subset: ', len(dataloader.sampler),  world_size, len(dataloader.dataset))
    
    
    return dataloader, aux_loader

def build_optimizer(optimizer, model):
    """
    Build optimizer, e.g., sgd, adamw.
    """
    assert optimizer.type in ["sgd", "adamw"], "Unknown optimizer type"

    # take parameters
    parameter_group_names = {"params": []}
    parameter_group_vars = {"params": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parameter_group_names["params"].append(name)
        parameter_group_vars["params"].append(param)

    # make optimizers
    if optimizer.type == "sgd":
        optimizer = optim.SGD(
            [parameter_group_vars], lr=optimizer.lr, momentum=optimizer.momentum, weight_decay=optimizer.weight_decay
        )
    elif optimizer.type == "adamw":
        optimizer = optim.AdamW(
            [parameter_group_vars], lr=optimizer.lr, weight_decay=optimizer.weight_decay
        )

    return optimizer

def build_scheduler(args, optimizer, n_iter_per_epoch, config=None):
    # modified later
    # num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    # warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    # decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    # multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]

    lr_scheduler = None
    if args.scheduler == "cycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr, epochs=args.total_epochs, steps_per_epoch=n_iter_per_epoch
        )
    elif args.scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )

    return lr_scheduler

def build_criterion(loss, alpha):
    if loss.type == "weak":
        criterion = WeakCEMatchLoss(loss)
    elif loss.type == "strong_ce":
        criterion = StrongCrossEntropyLoss(alpha)
    elif loss.type == "flow":
        criterion = StrongFlowLoss()
    else:
        raise ValueError("Unknown objective loss")

    return criterion

def build_checkpoint(model_dict, model, optimizer, lr_scheduler):
    max_pck = 0.0
    start_epoch = -100
    if model_dict.resume:
        Logger.info(f">>>>>>>>>> Resuming from {model_dict.resume} ..........")
        checkpoint = torch.load(model_dict.resume, map_location="cpu")

        msg = model.load_state_dict(checkpoint["model"], strict=False)
        Logger.info(msg)
        
        if (
            not model_dict.eval_mode  #TODO CHECK HERE
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1

            Logger.info(
                f"=> loaded successfully '{model_dict.resume}' (epoch {checkpoint['epoch']})"
            )

            if "max_pck" in checkpoint:
                max_pck = checkpoint["max_pck"]
            else:
                max_pck = 0.0

        del checkpoint
        torch.cuda.empty_cache()

    # load backbone
    if model_dict.backbone.pretrain in ["dino", "denseCL"]:
        Logger.info("Loading backbone from %s" % (model_dict.backbone.backbone_path))
        pretrained_backbone = torch.load(model_dict.backbone.backbone_path, map_location="cpu")
        backbone_keys = list(model.backbone.state_dict().keys())

        if "state_dict" in pretrained_backbone:
            model.load_backbone(pretrained_backbone["state_dict"], strict=False)
            load_keys = list(pretrained_backbone["state_dict"].keys())
        else:
            model.load_backbone(pretrained_backbone)
            load_keys = list(pretrained_backbone.keys())
        missing_keys = [i for i in backbone_keys if i not in load_keys]
        Logger.info("missing keys in loaded backbone: %s" % (missing_keys))

        del pretrained_backbone
        torch.cuda.empty_cache()

    return max_pck, start_epoch

def save_checkpoint(args, epoch, model, max_pck, optimizer, lr_scheduler):

    save_state = {'model': model.module.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'max_accuracy': max_pck,
                  'lr_scheduler': None,
                  'epoch': epoch,
                  'args': args}
    if lr_scheduler is not None:
        save_state['lr_scheduler'] = lr_scheduler.state_dict()

    save_path = os.path.join(args.logpath, f'ckpt_epoch_{epoch}.pth')
    Logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    Logger.info(f"{save_path} saved !!!")
    

