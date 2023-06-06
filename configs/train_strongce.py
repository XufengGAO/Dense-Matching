# Configuration file

batch_size = 16
w_group = 4
optimizer = dict(
    type='sgd',
    lr=1e-2,
    lr_backbone=0.0,
    weight_decay=1e-3,
    momentum=0.95
)
layers = [i for i in range(8, 17)]
use_wandb = True

# Training
start_epoch = 0
total_epochs = 50
scheduler = dict(
    type='none',
)
logpath = ''

# Datasets
data = dict(
    datapath='./datasets',
    alpha=0.1,
    batch_size=batch_size,
    train=dict(
        type='trn',
        benchmark='pfpascal',
        output_image_size = (240, 240),
        cam = '',
        thres = 'auto',
        sampler=dict(
            shuffle=True,
            drop_last=False,
        )
    ),
    val=dict(
        type='val',
        benchmark='pfpascal',
        output_image_size = (240, 240),
        cam = '',
        thres = 'auto',
        sampler=dict(
            shuffle=False,
            drop_last=True,
        )
    ),
    test=None
)

# Models
# r50 = [0] + [3, 4, 6, 3] = 17
# r101 = [0] + [3, 4, 23, 3] = 34
# layers = [3, 7, 13, 16]

init_type = 'kaiming_norm'

model = dict(
    backbone=dict(
        type='resnet',
        depth=50,
        pretrain='dino',
        backbone_path='./backbone/dino_resnet50.pth',
        layers=layers,
        freeze=True,
        cam=''
    ),
    
    use_neck = True,
    neck=dict(
        D=w_group,
        use_mp=False,
        init_type=init_type,
        use_relu=True
    ),

    use_head = False,
    head=dict(
        embed_dim=256,
        use_relu=True,
        init_type=init_type,
    ),

    resume = '',
    eval_mode = False,
)

# Loss
loss = dict(
    type='strong_ce',

    # arguments for weak loss
    temp=[10],
    match_loss_weight=0.1,
    ce_loss_weight=1.0,
    collect_grad=False,
    use_negative=True,
    match_layers=[i for i in range(4, 17)],
)

# Misc
wandb = dict(
    use=use_wandb,
    run_name = "%.e_D%d_m%.2f_bsz%d_Wd%.e_[%d,%d]_t%s_%s_%s" % \
      (optimizer['lr'], w_group, optimizer['momentum'], batch_size, \
       optimizer['weight_decay'], layers[0], layers[-1], loss['temp'], loss['type'], model['backbone']['pretrain']),
    run_id = '',
    proj_name = 'Multiple-W-Group',
    config = {
        'optimizer':optimizer['type'],
        'lr':optimizer['lr'],
        'lr_bone':optimizer['lr_backbone'],
        'Wd':optimizer['weight_decay'],
        'momentum':optimizer['momentum'],
        'bsz':batch_size,
        'D':w_group,
        'layers':"[%d,%d]"%(layers[0],layers[-1]),
        'pretrain':model['backbone']['pretrain'],
        'backbone':'%s%d'%(model['backbone']['type'],model['backbone']['depth']),
        'init':init_type,
        'scheduler':scheduler['type'],
        'alpha':data['alpha'],
        'img_size':data['train']['output_image_size'],
        'loss':loss['type'],
        'temp':loss['temp'],
        'use_negative':loss['use_negative']
        
    }
)

opm = dict(
    epsilon=0.05,
    exp2=0.5,
)


    # Training parameters
    # parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    # parser.add_argument("--scheduler", type=str, default="none", choices=['none', 'step', 'cycle', 'cosine'])
    # parser.add_argument('--step_size', type=int, default=16, help='hyperparameters for step scheduler')
    # parser.add_argument('--step_gamma', type=float, default=0.1, help='hyperparameters for step scheduler')

    # default is the value that the attribute gets when the argument is absent. const is the value it gets when given.

























