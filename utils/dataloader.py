import _init_paths

import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
from utils.data_augmentation import *

class video_data_loader(data.Dataset):
    def __init__(self, root, img_size, data_augmentation=False):
        self.img_size = img_size
        self.data_augmentation = data_augmentation
        self.images, self.flows, self.gts = [], [], []

        for seq_name in os.listdir(root):
            seq_flow = os.path.join(root, seq_name, 'Flow')
            seq_gt = os.path.join(root, seq_name, 'GT')
            self.flows += sorted([seq_flow + "/" + f for f in os.listdir(seq_flow) if f.endswith('.png') or f.endswith('jpg')])
            self.gts += sorted([seq_gt + '/' + f for f in os.listdir(seq_gt) if f.endswith('.jpg') or f.endswith('.png')])

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),  # bilinear
            transforms.ToTensor()])

        if self.data_augmentation:
            # self.resize = Resize(img_scale=(int(self.img_size*4), self.img_size), ratio_range=(0.5,2.0))
            # self.crop = RandomCrop(crop_size=(self.img_size, self.img_size), cat_max_ratio=0.8)
            self.flip = RandomFlip(prob=0.5)
            self.pmd = PhotoMetricDistortion()
            # self.pad = Pad(size=(self.img_size, self.img_size), pad_val=0, seg_pad_val=0)

        self.size = len(self.gts)

    def augmentation(self, result):
        # result = self.resize(result)
        # result = self.crop(result)
        result = self.flip(result)
        result = self.pmd(result)
        # result = self.pad(result)

        return result

    def __getitem__(self, index):
        flow_path = self.flows[index]
        gt_path = self.gts[index]
        img_path = gt_path.replace('GT', 'Frame')
        if os.path.exists(img_path) == False:
            img_path = img_path.replace('jpg', 'png')
        if os.path.exists(img_path) == False:
            img_path = img_path.replace('png', 'jpg')

        images = self.rgb_loader(img_path)
        flows = self.rgb_loader(flow_path)
        gts = self.binary_loader(gt_path)

        if self.data_augmentation:
            result = dict(img=np.array(images))
            result['flow'] = np.array(flows)
            result['gt_semantic_seg'] = np.array(gts)
            result['seg_fields'] = []
            result['seg_fields'].append('gt_semantic_seg')
            result = self.augmentation(result)

            images = Image.fromarray(result['img'])
            flows = Image.fromarray(result['flow'])
            gts = Image.fromarray(result['gt_semantic_seg'])

        images = self.img_transform(images)
        flows = self.img_transform(flows)
        gts = self.gt_transform(gts)
        # print(images.shape, flows.shape, gts.shape)

        return images, flows, gts, img_path

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root, use_flow):
        self.use_flow = use_flow
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))
        lst = []
        for name in lst_label:
            if name in lst_pred:
                lst.append(name)
        self.image_path = self.get_paths(lst, img_root)
        self.label_path = self.get_paths(lst, label_root)
        self.key_list = list(self.image_path.keys())

        self.check_path(self.image_path, self.label_path)
        self.trans = transforms.Compose([transforms.ToTensor()])


    def check_path(self, image_path_dict, label_path_dict):
        assert image_path_dict.keys() == label_path_dict.keys(), 'gt, pred must have the same videos'
        for k in image_path_dict.keys():
            assert len(image_path_dict[k]) == len(label_path_dict[k]), f'{k} have different frames'

    def get_paths(self, lst, root):
        v_lst = list(map(lambda x: os.path.join(root, x), lst))

        f_lst = {}
        for v in v_lst:
            v_name = v.split('/')[-1]
            if 'result' in root:
                if not self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:]
                elif self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:-1]  # 光流方法忽略第一帧和最后一帧

            elif 'TestSet' in root:
                if not self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, 'GT', f) for f in os.listdir(os.path.join(v, 'GT'))])[1:]
                elif self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, 'GT', f) for f in os.listdir(os.path.join(v, 'GT'))])[1:-1]  # 光流方法忽略第一帧和最后一帧
        return f_lst

    def read_picts(self, v_name):
        pred_names = self.image_path[v_name]
        pred_list = []
        for pred_n in pred_names:
            pred_list.append(self.trans(Image.open(pred_n).convert('L')))

        gt_names = self.label_path[v_name]
        gt_list = []
        for gt_n in gt_names:
            gt_list.append(self.trans(Image.open(gt_n).convert('L')))

        for gt, pred in zip(gt_list, pred_list):
            assert gt.shape == pred.shape, 'gt.shape!=pred.shape'
        
        gt_list = torch.cat(gt_list,dim=0)
        pred_list = torch.cat(pred_list,dim=0)
        return pred_list, gt_list

    def __getitem__(self, item):
        v_name = self.key_list[item]
        preds, gts = self.read_picts(v_name)

        return v_name, preds, gts

    def __len__(self):
        return len(self.image_path)
