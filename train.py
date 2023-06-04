import argparse
import os
import torch
import time
from model import DenseMatch
from model.postprocess import PostProcess
import tools.utils as utils
from tools.utils import AverageMeter, ProgressMeter
from tools.builder import init_distributed_mode, build_dataloader, build_optimizer, \
                            build_scheduler, build_criterion, build_checkpoint, save_checkpoint
from tools.logger import Logger
from tools.evaluation import Evaluator
import wandb
from mmcv import Config
import re
import torch.backends.cudnn as cudnn
from torch import distributed as dist
from tqdm import tqdm
wandb.login()


def train(cfg, model, criterion, dataloader, optimizer, epoch):
    # Logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

    # 1. Meters
    progress, meters = utils.get_Meters(cfg.loss.type, cfg.loss.collect_grad, len(dataloader), epoch)
    
    # 2. Model status
    model.module.backbone.eval()
    optimizer.zero_grad()
    running_total_loss = 0
    iters_per_epoch = len(dataloader)
    databar = tqdm(enumerate(dataloader), total=len(dataloader))
    for iter, data in databar:
        total_iters = iter + epoch * iters_per_epoch
        if cfg.use_wandb and dist.get_rank() == 0:
            wandb.log({'iters': total_iters})

        # 3. Get inputs
        bsz = data['src_img'].size(0)
        imgs, masks, num_negatives = utils.get_input(data, cfg.loss.use_negative)
        data["src_kps"] = data["src_kps"].cuda(non_blocking=True)
        data["n_pts"] = data["n_pts"].cuda(non_blocking=True)
        data["trg_kps"] = data["trg_kps"].cuda(non_blocking=True)
        data["pckthres"] = data["pckthres"].cuda(non_blocking=True)       

        # 4. Compute output
        src_feat, trg_feat, corr = model(imgs)

        with torch.no_grad():
            if evaluator.rf_center is None and evaluator.rf is None:
                rfsz, jsz, feat_h, feat_w, img_side = model.module.get_geo_info()
                evaluator.set_geo(rfsz, jsz, feat_h, feat_w)
        # 5. Calculate loss
        if cfg.loss.type == "strong_ce":
            with torch.no_grad():
                # return dict results                
                eval_result = evaluator.evaluate(data["src_kps"], data["trg_kps"], data["n_pts"],\
                                                 corr.detach().clone(), data['pckthres'], pck_only=False)
            loss = criterion(
                corr, eval_result['easy_match'], eval_result['hard_match'], data["pckthres"], data["n_pts"]
            )
            del eval_result
        elif cfg.loss.type == "weak":
            loss, ce_loss, match_loss = criterion(corr, src_feat, trg_feat, bsz, num_negatives)

            meters['EntropyLoss'].update(ce_loss, bsz)
            meters['MatchLoss'].update(match_loss, bsz)

        del corr

        meters['loss'].update(loss.item(), bsz)

        # 6. BP
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        
        # 7. print running pck, loss (iter % 100 == 0) and 
        running_total_loss += loss.item()
        if dist.get_rank() == 0:
            # progress.display(iter+1)
            databar.set_description(
                'Training [%d/%d]: R_total_loss: %.3f/%.3f' % (epoch, cfg.total_epochs, running_total_loss/(iter + 1), loss.item()))

        del src_feat, trg_feat, data, loss
    # torch.cuda.empty_cache()

    # 8. collect gradients
    dist.barrier()
    for loss_name in list(meters.keys()):
        meters[loss_name].all_reduce()

    if cfg.loss.type == "weak" and cfg.use_wandb and dist.get_rank() == 0:
            wandb.log({"EntropyLoss": meters['EntropyLoss'].avg, "MatchLoss": meters['MatchLoss'].avg})
    
    return meters['loss'].avg

def validate(cfg, model, criterion, dataloader, epoch, aux_val_loader=None):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            databar = tqdm(enumerate(loader), total=len(loader))
            for _, data in databar:
                # 1. Get inputs
                bsz = data["src_img"].size(0)
                imgs, masks, num_negatives = utils.get_input(data, use_negative=False)
                data["src_kps"] = data["src_kps"].cuda(non_blocking=True)
                data["n_pts"] = data["n_pts"].cuda(non_blocking=True)
                data["trg_kps"] = data["trg_kps"].cuda(non_blocking=True)
                data["pckthres"] = data["pckthres"].cuda(non_blocking=True)                

                # 2. Compute output
                src_feat, trg_feat, corr = model(imgs, bsz)

                # 3. Calculate loss
                if cfg.loss.type == "strong_ce":
                    # return dict results                
                    eval_result = evaluator.evaluate(data["src_kps"], data["trg_kps"], data["n_pts"],\
                                                    corr.detach().clone(), data['pckthres'], pck_only=False)
                    loss = criterion(
                        corr, eval_result['easy_match'], eval_result['hard_match'], data["pckthres"], data["n_pts"]
                    )
                    del eval_result
                elif cfg.loss.type == "weak":
                    loss, _, _ = criterion(corr, src_feat, trg_feat, bsz, num_negatives)

                loss_meter.update(loss.item(), bsz)

                # 4. Calculate PCK
                ##TODO, optmatch should be added later, Post-processing
                with torch.no_grad():
                    geometric_scores = torch.stack([PostProcessModule.rhm(c.clone().detach()) for c in corr], dim=0)
                corr *= geometric_scores

                pck = evaluator.evaluate(data["src_kps"], data["trg_kps"], data["n_pts"],\
                                            corr.detach().clone(), data['pckthres'], pck_only=True)

                batch_pck = utils.mean(pck)

                pck_meter.update(batch_pck, bsz)
   
                del batch_pck, data

    loss_meter = AverageMeter('loss', ':4.2f')
    pck_meter = AverageMeter('pck', ':4.2f')
    progress_list = [loss_meter, pck_meter]
    progress = ProgressMeter(
        len(dataloader) + (cfg.distributed and (len(dataloader.sampler) * cfg.world_size < len(dataloader.dataset))), 
        progress_list, prefix="Validation".format(epoch))

    # switch to evaluate mode
    model.module.backbone.eval()

    run_validate(dataloader)

    dist.barrier()
    loss_meter.all_reduce()
    pck_meter.all_reduce()

    if aux_val_loader is not None:
        run_validate(aux_val_loader, len(dataloader))

    dist.barrier()
    progress.display_summary()

    avg_loss = loss_meter.avg
    avg_pck = pck_meter.avg

    return avg_loss, avg_pck

def build_wandb(wandb_dict, rank):
    if wandb_dict.use and rank == 0:

        _ = wandb.init(
            project=wandb_dict.proj_name,
            config=wandb_dict.config,
            id=wandb_dict.run_id,
            resume="allow",
            name=wandb_dict.run_name,
        )

        wandb.define_metric("iters")

        wandb.define_metric("epochs")
        wandb.define_metric("trn_loss", step_metric="epochs")

        wandb.define_metric("val_loss", step_metric="epochs")
        wandb.define_metric("val_pck", step_metric="epochs")
        
        wandb.define_metric("EntropyLoss", step_metric="epochs")
        wandb.define_metric("MatchLoss", step_metric="epochs")             

def main(cfg):
    # 1. Init Logger
    cfg.wandb.config['log'] = Logger.initialize(cfg)

    rank = dist.get_rank()
    local_rank = cfg.local_rank
    device = torch.device("cuda:{}".format(local_rank))
    world_size = dist.get_world_size()

    utils.fix_randseed(seed=0)
    cudnn.benchmark = True
    # cudnn.deterministic = True
    
    # 2. Dataloader
    train_loader, _ = build_dataloader(cfg.data.train, cfg.data.datapath, cfg.data.batch_size, rank, world_size)
    val_loader, aux_val_loader = build_dataloader(cfg.data.val, cfg.data.datapath, cfg.data.batch_size, rank, world_size)

    # 3. Model
    model = DenseMatch.Model(cfg.model, cfg.loss)
    model.cuda()

    # freeze the backbone
    if model.freeze:
        Logger.info(f'>>>>>>>>>> Backbone frozen!')
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

    # 4. Optimizer, Scheduler, and Loss
    optimizer = build_optimizer(cfg.optimizer, model)
    # To give scheduler
    #lr_scheduler = build_scheduler(args, optimizer, len(train_loader), config=None)
    lr_scheduler = None
    criterion = build_criterion(cfg.loss, cfg.data.alpha)

    # 5. Distributed training
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[cfg.local_rank],
        output_device=cfg.local_rank,
        find_unused_parameters=False,
    )
    
    # check # of training params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger.info(f">>>>>>>>>> Number of training params: {n_parameters}")

    # resume the model
    if cfg.model.resume or cfg.model.backbone.pretrain in ["dino", "denseCL"]:
        model_without_ddp = model.module
        max_pck, start_epoch = build_checkpoint(cfg.model, model_without_ddp, optimizer, lr_scheduler)
        if start_epoch > 0:
            cfg.start_epoch = start_epoch
    else:
        max_pck = 0.0

    # 6. Evaluator, Wandb
    global evaluator
    evaluator = Evaluator(criterion=cfg.loss.type, device=device, alpha=cfg.data.alpha)
    build_wandb(cfg.wandb, rank)

    # 7. Start training
    log_benchmark = {}
    Logger.info(">>>>>>>>>> Start training")

    global PostProcessModule
    PostProcessModule = None

    dist.barrier()
    for epoch in range(cfg.start_epoch, cfg.total_epochs):
        train_loader.sampler.set_epoch(epoch)

        # train
        start_time = time.time()
        trn_loss = train(cfg, model, criterion, train_loader, optimizer, epoch)
        log_benchmark["trn_loss"] = trn_loss
        end_train_time = (time.time()-start_time)/60

        # Create a PostProcessModule:
        if PostProcessModule is None:
            rfsz, jsz, feat_h, feat_w, img_side = model.module.get_geo_info()
            PostProcessModule = PostProcess(rfsz, jsz, feat_h, feat_w, device, img_side)

        # validation
        start_time = time.time()
        val_loss, val_pck = validate(
            cfg, model, criterion, val_loader, epoch, aux_val_loader
        )
        log_benchmark["val_loss"] = val_loss
        log_benchmark["val_pck"] = val_pck
        end_val_time = (time.time()-start_time)/60

        # save model and log results
        if val_pck > max_pck and rank == 0:
            # Logger.save_model(model.module, epoch, val_pck, max_pck)
            # save_checkpoint(args, epoch, model, max_pck, optimizer, lr_scheduler)
            Logger.info('Best Model saved @%d w/ val. PCK: %5.4f -> %5.4f on [%s]' % (epoch, max_pck, val_pck, os.path.join(cfg.logpath, f'ckpt_epoch_{epoch}.pth')))
            max_pck = val_pck

        if cfg.wandb.use and rank == 0:
            wandb.log({"epochs": epoch})
            wandb.log(log_benchmark)
            
        time_message = (
            ">>>>> Train/Eval %d epochs took:%4.3f + %4.3f = %4.3f"%(epoch + 1, end_train_time, end_val_time, end_train_time+end_val_time)+" minutes\n"
        )
        Logger.info(time_message)
        # if epoch%2 == 0:
        #     torch.cuda.empty_cache()

    Logger.info("==================== Finished training ====================")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training Script")    
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='numer of distributed processes')
    parser.add_argument("--local_rank", required=True, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

    
    args = parser.parse_args()
    
    dist_dict = init_distributed_mode(args)

    if args.config:
        dist_dict['config'] = args.config
        cfg = Config.fromfile(args.config)
        cfg.merge_from_dict(dist_dict)
    else:
        raise ValueError('Config is required')

    if cfg.loss.type in ['strong_ce']:
        cfg.loss.use_negative = False

    if cfg.wandb.use and cfg.wandb.run_id == '':
        cfg.wandb.run_id = wandb.util.generate_id()
    
    main(cfg)
