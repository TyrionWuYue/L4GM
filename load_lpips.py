import os
import torch
from kiui.lpips import LPIPS

os.environ['TORCH_HOME'] = '/Users/wuyue/Desktop/L4GM/checkpoints'
lpips = LPIPS(net='vgg')