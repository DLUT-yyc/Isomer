import os
import _init_paths
from config import Parameters
opt = Parameters().parse()

import time
import torch
from torch.utils import data
from lib.builder import VOSNet

from utils.dataloader import video_data_loader
from utils.init import load_model
from utils.func import save_images


def demo(opt):

    model = VOSNet(opt)
    model = load_model(model, opt)

    model.cuda()
    model.eval()
    test_dataset_list = opt.infer_dataset
    for dataset in test_dataset_list.split(','):
        save_path = opt.infer_save + dataset + '/'

        test_dataset = video_data_loader(opt.infer_dataset_path + dataset, img_size=opt.img_size, data_augmentation=False)

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=opt.val_batchsize, shuffle=False, num_workers=32, pin_memory=True)

        dataset_size = len(test_dataset)

        with torch.no_grad():
            img_count = 1
            time_total = 0
            for step, data_pack in enumerate(test_loader):

                images, flows, gts, img_paths = data_pack
                images = images.cuda()
                flows = flows.cuda()

                bs, _, _, _ = images.size()

                time_start = time.perf_counter()
                sals  = model(images, flows)
                cur_time = (time.perf_counter() - time_start)

                time_total += cur_time

                for index in range(bs):
                    sal = sals[index, :, :, :].unsqueeze(0)
                    save_images(sal, img_paths, index, save_path, opt)

                    print('[INFO-Test] Dataset: {}, Image: ({}/{}), '
                          'TimeCom: {}'.format(dataset, img_count, dataset_size, cur_time / bs))
                    img_count += 1
            print("\n[INFO-Test-Done] FPS: {}".format(dataset_size / time_total))

if __name__ == "__main__":
    demo(opt=opt)
