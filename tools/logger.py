r"""Logging"""
import logging
import os
import torch
from torch import distributed as dist

class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, args, training=True):
        if training:
            if args.logpath == "":
                logpath = "%.e_%s_%s_bsz%d"%(args.lr, args.loss_stage, args.optimizer, args.batch_size)
                
                if args.optimizer == "sgd":
                    logpath = logpath + "_m%.2f"%(args.momentum)
                if args.scheduler != 'none':
                    logpath = logpath + "_%s"%(args.scheduler)

                cls.logpath = os.path.join('logs', 'ddp', 'train', args.backbone, args.pretrain, args.criterion, args.benchmark + "_%s"%(args.alpha), logpath)
                filemode = 'w'
            else:
                cls.logpath = args.logpath
                filemode = 'a'
        else:
            # logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
            cls.logpath = os.path.join('logs', 'test', args.backbone, args.pretrain, args.criterion, args.benchmark, args.logpath)
            filemode = 'w'
        
        if dist.get_rank() == 0:
            os.makedirs(cls.logpath, exist_ok=True)
        dist.barrier()
  

        logging.basicConfig(filemode=filemode,
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO if dist.get_rank()==0 else logging.WARN,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')
        # if dist.get_rank()==0 else logging.WARN
        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
#         cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))
        
        # Log arguments
        logging.info('\n+=========== Arguments ============+')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
        logging.info('+================================================+\n')

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        logging.info(msg)

    @classmethod
    def save_epoch(cls, model, epoch, val_pck):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'eooch_%d.pt'%(epoch)))
        cls.info('Epoch Model saved @%d w/ val. PCK: %5.4f on [%s]\n' % (epoch, val_pck, os.path.join(cls.logpath, 'eooch_%d.pt'%(epoch))))
    
    @classmethod
    def save_model(cls, model, epoch, val_pck, best_val_pck):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Best Model saved @%d w/ val. PCK: %5.4f -> %5.4f on [%s]\n' % (epoch, best_val_pck, val_pck, os.path.join(cls.logpath, 'best_model.pt')))


























