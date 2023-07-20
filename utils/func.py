import _init_paths

import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict


def save_images(sal, img_paths, bs_index, save_path, opt):
    tmp = img_paths[bs_index].split('/')
    os.makedirs(os.path.join(save_path, tmp[-3]), exist_ok=True)
    sal_name = tmp[-3] + '/' + tmp[-1].replace('.jpg', '.png')
    # sal_name = tmp[-3] + '/' + tmp[-1][-9:].replace('.jpg', '.png')     # python_val index.png

    gt = Image.open(img_paths[bs_index])
    gt = np.asarray(gt, np.float32)
    # gt /= (gt.max() + 1e-8)

    sal = F.interpolate(sal, size=(gt.shape[0], gt.shape[1]), mode='bilinear', align_corners=False)
    sal = sal.sigmoid().data.cpu().numpy().squeeze()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    
    # For ZVOS
    # sal[np.where(sal>=0.5)] = 1
    # sal[np.where(sal<0.5)] = 0

    sal = (sal*255).astype(np.uint8)

    sal = Image.fromarray(sal)
    sal.save(save_path + '/' + sal_name)

def rm_module(saved_state_dict):
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict['network'].items():
        new_state_dict[k.replace('module.', '')] = v
    return new_state_dict



