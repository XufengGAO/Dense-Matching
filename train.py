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


def train(args, model, criterion, dataloader, optimizer, epoch):
    # Logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

    # 1. Meters
    progress, meters = utils.get_Meters(args.criterion, args.collect_grad, len(dataloader), epoch)
    
    # 2. Model status
    model.module.backbone.eval()
    optimizer.zero_grad()
    running_total_loss = 0
    iters_per_epoch = len(dataloader)
    databar = tqdm(enumerate(dataloader), total=len(dataloader))
    for iter, data in databar:
        total_iters = iter + epoch * iters_per_epoch
        if args.use_wandb and dist.get_rank() == 0:
            wandb.log({'iters': total_iters})

        # 3. Get inputs
        bsz = data['src_img'].size(0)
        imgs, masks, num_negatives = utils.get_input(data, args.use_negative)
        data["src_kps"] = data["src_kps"].cuda(non_blocking=True)
        data["n_pts"] = data["n_pts"].cuda(non_blocking=True)
        data["trg_kps"] = data["trg_kps"].cuda(non_blocking=True)
        data["pckthres"] = data["pckthres"].cuda(non_blocking=True)       

        # 4. Compute output
        src_feat, trg_feat, corr = model(imgs, bsz)

        with torch.no_grad():
            if evaluator.rf_center is None and evaluator.rf is None:
                rfsz, jsz, feat_h, feat_w, img_side = model.module.get_geo_info()
                evaluator.set_geo(rfsz, jsz, feat_h, feat_w)
        # 5. Calculate loss
        if args.criterion == "strong_ce":
            with torch.no_grad():
                # return dict results                
                eval_result = evaluator.evaluate(data["src_kps"], data["trg_kps"], data["n_pts"],\
                                                 corr.detach().clone(), data['pckthres'], pck_only=False)
            loss = criterion(
                corr, eval_result['easy_match'], eval_result['hard_match'], data["pckthres"], data["n_pts"]
            )
            del eval_result
        elif args.criterion == "weak":
            task_loss = criterion(corr, src_feat, trg_feat, bsz, num_negatives)

            meters['EntropyLoss'].update(task_loss[0].item(), bsz)
            meters['MatchLoss'].update(task_loss[1].item(), bsz)

            loss = (weak_lambda * task_loss).sum()

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
                'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (iter + 1), loss.item()))

        del src_feat, trg_feat, data, loss
    # torch.cuda.empty_cache()

    # 8. collect gradients
    dist.barrier()
    for loss_name in list(meters.keys()):
        meters[loss_name].all_reduce()

    if args.criterion == "weak" and args.use_wandb and dist.get_rank() == 0:
            wandb.log({"EntropyLoss": meters['EntropyLoss'].avg, "MatchLoss": meters['MatchLoss'].avg})
    
    return meters['loss'].avg

def validate(args, model, criterion, dataloader, epoch, aux_val_loader=None):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            for _, data in enumerate(loader):
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
                if args.criterion == "strong_ce":
                    # return dict results                
                    eval_result = evaluator.evaluate(data["src_kps"], data["trg_kps"], data["n_pts"],\
                                                    corr.detach().clone(), data['pckthres'], pck_only=False)
                    loss = criterion(
                        corr, eval_result['easy_match'], eval_result['hard_match'], data["pckthres"], data["n_pts"]
                    )
                    del eval_result
                elif args.criterion == "weak":
                    task_loss = criterion(corr, src_feat, trg_feat, bsz, num_negatives)
                    loss = (weak_lambda * task_loss).sum()

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
        len(dataloader) + (args.distributed and (len(dataloader.sampler) * args.world_size < len(dataloader.dataset))), 
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


def build_wandb(args, rank):
    if args.use_wandb and rank == 0:
        wandb_name = "%.e_%s_%s" % (
            args.lr,
            args.criterion,
            args.optimizer,
        )
        if args.scheduler != "none":
            wandb_name += "_%s" % (args.scheduler)
        if args.optimizer == "sgd":
            wandb_name = wandb_name + "_m%.2f" % (args.momentum)
        
        if args.use_negative:
            wandb_name += "_bsz%d-neg" % (args.batch_size*2)
            args.batch_size = args.batch_size*2
        else:
            wandb_name += "_bsz%d" % (args.batch_size)

        if args.criterion == 'weak':
            wandb_name += ("_%s_tp%.2f"%(args.weak_lambda, args.temp))

        _wandb = wandb.init(
            project=args.wandb_name,
            config=args,
            id=args.run_id,
            resume="allow",
            name=wandb_name,
        )

        wandb.define_metric("iters")
        wandb.define_metric("running_avg_loss", step_metric="iters")
        wandb.define_metric("running_avg_pck", step_metric="iters")

        wandb.define_metric("epochs")
        wandb.define_metric("trn_loss", step_metric="epochs")
        wandb.define_metric("trn_pck", step_metric="epochs")

        wandb.define_metric("val_loss", step_metric="epochs")
        wandb.define_metric("val_pck", step_metric="epochs")

        if args.criterion == "weak":
            wandb.define_metric("discSelf_loss", step_metric="epochs")
            wandb.define_metric("discCross_loss", step_metric="epochs")
            wandb.define_metric("match_loss", step_metric="epochs")
            
            if args.collect_grad:
                wandb.define_metric("discSelfGrad", step_metric="iters")
                wandb.define_metric("discCrossGrad", step_metric="iters")
                wandb.define_metric("matchGrad", step_metric="iters")                

def main(args):
    # 1. Init Logger
    Logger.initialize(args, training=True)
    args.logpath = Logger.logpath

    rank = dist.get_rank()
    local_rank = args.local_rank
    device = torch.device("cuda:{}".format(local_rank))
    world_size = dist.get_world_size()

    utils.fix_randseed(seed=0)
    cudnn.benchmark = True
    # cudnn.deterministic = True
    
    # 2. Dataloader
    train_loader, val_loader, aux_val_loader = build_dataloader(args, rank, world_size)

    # 3. Model
    assert args.backbone in ["resnet50", "resnet101"], "Unknown backbone"
    Logger.info(f">>>>>>>>>> Creating model:{args.backbone} + {args.pretrain} <<<<<<<<<<")

    if args.layers:
        layers = args.layers
    else:
        n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
        layers = list(range(n_layers[args.backbone]))
    str_layer = ', '.join(str(i) for i in layers)
    Logger.info(f'>>>>>>> Use {len(layers)} layers: {str_layer}.')

    #TODO: directly change here
    model = DenseMatch.Model(args, layers, freeze=args.freeze_backbone)
    model.cuda()

    # freeze the backbone
    if args.freeze_backbone:
        Logger.info(f'>>>>>>>>>> Backbone frozen!')
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

    # 4. Optimizer, Scheduler, and Loss
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_scheduler(args, optimizer, len(train_loader), config=None)
    criterion = build_criterion(args)

    # 5. Distributed training
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False,
    )
    
    # check # of training params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger.info(f">>>>>>>>>> Number of training params: {n_parameters}")

    # resume the model
    if args.resume or args.pretrain in ["dino", "denseCL"]:
        model_without_ddp = model.module
        max_pck = build_checkpoint(args, model_without_ddp, optimizer, lr_scheduler)
    else:
        max_pck = 0.0

    # 6. Evaluator, Wandb
    global evaluator
    evaluator = Evaluator(criterion=args.criterion, device=device, alpha=args.alpha)
    build_wandb(args, rank)

    # 7. Start training
    log_benchmark = {}
    Logger.info(">>>>>>>>>> Start training")
    if args.criterion == 'weak':
        global weak_lambda
        weak_lambda = torch.FloatTensor(list(map(float, re.findall(r"[-+]?(?:\d*\.*\d+)", args.weak_lambda)))).cuda()
        weak_lambda.requires_grad = False

    global PostProcessModule
    PostProcessModule = None

    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        # train
        start_time = time.time()
        trn_loss = train(args, model, criterion, train_loader, optimizer, epoch)
        log_benchmark["trn_loss"] = trn_loss
        end_train_time = (time.time()-start_time)/60

        # Create a PostProcessModule:
        if PostProcessModule is None:
            rfsz, jsz, feat_h, feat_w, img_side = model.module.get_geo_info()
            PostProcessModule = PostProcess(rfsz, jsz, feat_h, feat_w, device, img_side)

        # validation
        start_time = time.time()
        val_loss, val_pck = validate(
            args, model, criterion, val_loader, epoch, aux_val_loader
        )
        log_benchmark["val_loss"] = val_loss
        log_benchmark["val_pck"] = val_pck
        end_val_time = (time.time()-start_time)/60

        # save model and log results
        if val_pck > max_pck and rank == 0:
            # Logger.save_model(model.module, epoch, val_pck, max_pck)
            # save_checkpoint(args, epoch, model, max_pck, optimizer, lr_scheduler)
            Logger.info('Best Model saved @%d w/ val. PCK: %5.4f -> %5.4f on [%s]' % (epoch, max_pck, val_pck, os.path.join(args.logpath, f'ckpt_epoch_{epoch}.pth')))
            max_pck = val_pck

        if args.use_wandb and rank == 0:
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
    # Arguments parsing
    # fmt: off
    parser = argparse.ArgumentParser(description="Training Script")
    
    parser.add_argument('--config', help='train config file path')

    # Datasets
    parser.add_argument('--datapath', type=str, default='./datasets') 
    parser.add_argument('--benchmark', type=str, default='pfpascal')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--output_image_size', type=str, default='(300)')
    parser.add_argument('--classmap', type=int, default=0, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')

    # Models
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrain', type=str, default='imagenet', choices=['imagenet', 'dino', 'denseCL'], help='supervised or self-supervised backbone')
    parser.add_argument('--backbone_path', type=str, default='./backbone')
    parser.add_argument('--layers', type=str, default='')
    parser.add_argument('--freeze_backbone', type= utils.boolean_string, nargs="?", default=True)
    
    # Custom module
    parser.add_argument("--init_type", type= str, nargs="?", default='kaiming_norm')
    parser.add_argument('--w_group', type=int, default=1)
    parser.add_argument('--use_mp', type= utils.boolean_string, nargs="?", default=True)
    parser.add_argument('--embed_dim', type=int, default=256, help='dimension of projection')
    parser.add_argument('--use_feat_project', type= utils.boolean_string, nargs="?", default=False)
    
    # Training parameters
    # parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--lr_backbone', type=float, default=0.0) 
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay (default: 0.00)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument("--scheduler", type=str, default="none", choices=['none', 'step', 'cycle', 'cosine'])
    parser.add_argument('--step_size', type=int, default=16, help='hyperparameters for step scheduler')
    parser.add_argument('--step_gamma', type=float, default=0.1, help='hyperparameters for step scheduler')

    # Misc
    parser.add_argument("--use_wandb", type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
    parser.add_argument('--run_id', type=str, default='', help='run_id')
    parser.add_argument('--wandb_name', type=str, default='', help='wandb project name')


    # SCOT algorithm parameters
    # parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    # parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    # parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    # parser.add_argument('--epsilon', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')

    # default is the value that the attribute gets when the argument is absent. const is the value it gets when given.


    # Loss parameters
    parser.add_argument('--criterion', type=str, default='strong_ce', choices=['weak', 'strong_ce', 'flow'])
    parser.add_argument('--weak_lambda', type=str, default='[1.0, 1.0]')
    parser.add_argument('--temp', type=float, default=0.05, help='softmax-temp for match loss')
    parser.add_argument("--collect_grad", type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument("--use_negative", type= utils.boolean_string, nargs="?", default=False)


    # Arguments for distributed data parallel
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='numer of distributed processes')
    parser.add_argument("--local_rank", required=True, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

    
    args = parser.parse_args()
    
    dist_dict = init_distributed_mode(args)
    args.output_image_size = utils.parse_string(args.output_image_size)

    if args.config:
        dist_dict['config'] = args.config
        args = Config.fromfile(args.config)
        args.merge_from_dict(dist_dict)

    if args.use_negative:
        assert args.criterion == 'weak', "use_negative is only for weak loss"

    if args.use_wandb and args.run_id == '':
        args.run_id = wandb.util.generate_id()
    
    main(args)
