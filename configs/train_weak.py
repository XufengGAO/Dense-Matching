# Configuration file

# Datasets
datapath = './datasets'
benchmark = "pfpascal"
alpha = 0.1
output_image_size = (256, 256)
cam = ''
classmap = 0

# Models
backbone = "resnet50"
pretrain = 'imagenet' 
backbone_path = './backbone/dino_resnet50.pth'
# r50 = [0] + [3, 4, 6, 3] = 17
# r101 = [0] + [3, 4, 23, 3] = 34
layers = [i for i in range(8, 17)]
freeze_backbone = True

# Custom module
init_type = 'kaiming_norm'
w_group = 2
use_mp = True # matrix_product
embed_dim = 256
use_feat_project = False


# Training
lr = 0.001
epochs = 100
start_epoch = 0
batch_size = 8 
optimizer = 'sgd'
weight_decay = 0.0001
momentum = 0.9
scheduler = 'none'


# Misc
use_wandb = True
wandb_name = 'ddp_scot'
run_id = ''
logpath = ''


# Loss
criterion = 'strong_ce'
weak_lambda = '[1.0, 0.0]'
temp = 0.05 
collect_grad = False
use_negative = False 























