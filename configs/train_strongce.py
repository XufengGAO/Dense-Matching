# Configuration file

batch_size = 4
w_group = 16
optimizer = dict(
    type='sgd',
    lr=1e-4,
    lr_backbone=0.0,
    weight_decay=1e-5,
    momentum=0.95
)

# Datasets
data = dict(
    datapath='./datasets',
    alpha=0.1,
    batch_size=batch_size,
    train=dict(
        type='trn',
        benchmark='pfpascal',
        output_image_size = (200, 300),
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
        output_image_size = (200, 300),
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
layers = [i for i in range(4, 17)]
init_type = 'xavier_norm'

model = dict(
    backbone=dict(
        type='resnet',
        depth=50,
        pretrain='imagenet',
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
        use_relu=False
    ),

    use_head = False,
    head=dict(
        embed_dim=256,
        use_relu=True,
        init_type=init_type,
    ),

    resume = False,
    eval_mode = False,
)

# Loss
loss = dict(
    type='strong_ce',

    # arguments for weak loss
    temp=[0.05],
    match_loss_weight=0.0,
    ce_loss_weight=0.0,
    collect_grad=False,
    use_negative=False,
    match_layers=[i for i in range(8, 17)],
)



# Training
total_epochs = 50
start_epoch = 0
scheduler = 'none'

# Misc
use_wandb = True
logpath = ''
resume = ''
run_id = ''
wandb_proj_name = 'Multiple-W-Group'

























