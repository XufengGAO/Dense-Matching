r"""Logging"""
import logging
import os
from torch import distributed as dist
import datetime

class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, cfg):

        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        cls.logpath = os.path.join('logs', logtime + '.log')

        if dist.get_rank() == 0:
            os.makedirs(cls.logpath, exist_ok=True)
        dist.barrier()
  

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO if dist.get_rank()==0 else logging.WARN,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
#         cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))
        
        if dist.get_rank()==0:
            cfg.dump(os.path.join(cls.logpath, os.path.basename(cfg.config)))
        
        return cls.logpath

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        logging.info(msg)


























