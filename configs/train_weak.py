# Configuration file

# Datasets
datapath = './datasets'
benchmark = "pfpascal"
alpha = 0.1
thres = 'auto'
output_image_size = (200, 300)
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
init_type = 'xavier_norm'
w_group = 1
use_mp = False # matrix_product

embed_dim = 128
use_feat_project = False

# Training
lr = 0.001
lr_backbone = 0.0
epochs = 200
start_epoch = 0
batch_size = 16
optimizer = 'sgd'
weight_decay = 0.00001
momentum = 0.9
scheduler = 'none'

# Misc
use_wandb = False
logpath = ''
resume = ''
run_id = ''
wandb_name = 'Multiple-W-Group'

# Loss
criterion = 'weak'
weak_lambda = '[1.0, 0.0]'
temp = 0.05 
collect_grad = False
use_negative = True 


























