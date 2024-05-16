import torch
import time
import argparse
from model import fusion_refine,Discriminator
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset
from torch.utils.data import DataLoader
import os
from torchvision.models import vgg16
from utils_test import to_psnr,to_ssim_skimage
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
# hyper-parameters train #
parser = argparse.ArgumentParser()
parser.add_argument('learning_rate', default=0.00001, type=float)
parser.add_argument('train_batch_size', default= 20, type=int)
parser.add_argument('train_epoch', default=1e4, type=int)
parser.add_argument('train_dataset', default='', type=str)
parser.add_argument('data_dir', default='', type=str)
parser.add_argument('model_save_dir', default='', type=str)
parser.add_argument('log_dir', default='', type=str)
# parameters test #
parser.add_argument('test_dataset', default='', type=str)
parser.add_argument('predict_result', default='', type=str)
parser.add_argument('test_batch_size', default=1, type=int)
parser.add_argument('vgg_model', default='', type=str)
parser.add_argument('imagenet_model', default='', type=str)
parser.add_argument('rcan_model', default='', type=str)
args = parser.parse_args()
### hyper-parameters over
### used for train, Instantiation parameter
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
train_dataset = args.train_dataset

# used for test #
test_dataset = args.test_dataset
predict_result = args.predict_result
test_batch_size = args.test_batch_size

# 判断模型路径是否存在
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir = os.path.join(args.model_save_dir, 'output_result')

#定义运行环境
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = troch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#定义主要网络
Net = fusion_refine(args.imagenet_model, args.rcan_model)
#现实网络中现有的参数量
print('our main model include parameters:', sum(param.numel() for param in Net.parameters()))
#定义判别器网络
DNet = Discriminator()
#显示判别器中有的参数量
print('our discriminator model', sum(param.numel() for param in DNet.parameters()))

#定义优化器
G_optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
scherduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[2000,5000,8000], gamma = 0.1)
D_optimizer = torch.optim.Adam(Net.parameters(), lr=0.0001)
scherduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optimizer, milestones=[2000,5000,8000], grmma = 0.5)

#定义训练数据






