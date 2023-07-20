import _init_paths

import os
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.autograd import Variable

from utils.func import *
import torch.nn.functional as F
from utils.evaluator import Eval_thread
from utils.dataloader import EvalDataset
from utils.val_zvos import compute_J_F

def train_one_epoch(model, loss_func, optimizer, data_loader, sampler, writer, epoch, iters, total_iters, iters_pre_epoch, opt):
    model.train()
    optimizer.zero_grad()
    sampler.set_epoch(epoch)
    # ---- multi-scale training ----
    size_rates = [0.75,1,1.25] if opt.ms_train == True else [1]

    loop = tqdm(enumerate(data_loader, start=1), total =len(data_loader))
    for i, pack in loop:
        iters += 1
        for rate in size_rates:
            # ---- get data ----
            images, flows, gts, _img_path = pack
            images = Variable(images.cuda())
            flows = Variable(flows.cuda())
            gts = Variable(gts.cuda())
            # ---- multi-scale training ----
            img_size = int(round(opt.img_size*rate/32)*32)

            if rate != 1:
                images = F.interpolate(images, size=(img_size, img_size), mode='bilinear', align_corners=False)
                gts = F.interpolate(gts, size=(img_size, img_size), mode='nearest')

            # ---- forward ----
            preds = model(images, flows)
            # ---- cal loss ----
            loss = loss_func(preds, gts)
            # ---- backward ----
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            total_loss = loss / dist.get_world_size()

        if opt.local_rank == 0:
            writer.add_scalar('train_loss', total_loss, iters)
        loop.set_description(f'Epoch [{epoch}/{opt.epoch}] Training   [Video]')
        loop.set_postfix(loss = total_loss.item(), lr = optimizer.param_groups[0]['lr'])

    return iters

def val_one_epoch(model, val_data_loader, writer, epoch, opt, val_dataset):
    model.eval()
    with torch.no_grad():
        img_save_path = os.path.join(opt.infer_save, opt.trainset, val_dataset)
        loop = tqdm(enumerate(val_data_loader), total =len(val_data_loader))
        for index, batch in loop:
            img, flow, gt, img_paths = batch
            pred = model.module(Variable(img.cuda()), Variable(flow.cuda()))
            for bs_index in range(img.size()[0]):
                sal = pred[bs_index, :, :, :].unsqueeze(0)
                save_images(sal, img_paths, bs_index, img_save_path, opt)
            loop.set_description(f'Epoch [{epoch}/{opt.epoch}] Inference  [{val_dataset}]')

    if val_dataset == 'MCL':
        loader = EvalDataset(img_root=img_save_path, label_root='../dataset/TestSet/'+val_dataset+'/', use_flow=True)
        thread = Eval_thread(loader, opt.trainset, val_dataset)
        measure_results = thread.run(epoch, opt.epoch)
        writer.add_scalar(val_dataset + '_MAE', measure_results[0][0], epoch)
        writer.add_scalar(val_dataset + '_Max_Fmeasure', measure_results[0][1], epoch)
        writer.add_scalar(val_dataset + '_Max_Emeasure', measure_results[0][2], epoch)
        writer.add_scalar(val_dataset + '_Max_Smeasure', measure_results[0][3], epoch)
        return 0
    elif val_dataset in ['DAVIS', 'FBMS', 'Long_Videos']:
        J_mean, F_mean = compute_J_F(epoch, opt.epoch, val_dataset, opt.val_root, os.path.join(opt.infer_save, opt.trainset))
        writer.add_scalar(val_dataset + '_J', J_mean, epoch)
        writer.add_scalar(val_dataset + '_F', F_mean, epoch)
        JF_mean = (J_mean + F_mean) / 2
        writer.add_scalar(val_dataset + '_J&F', JF_mean, epoch)
        return JF_mean

def save_model_optimizer(model, optimizer, epoch, opt, save_best):
    os.makedirs(opt.save_path, exist_ok=True)
    state = {'epoch':epoch, 'network':model.state_dict(), 'optimizer':optimizer.state_dict()}
    if save_best:
        torch.save(state, opt.save_path + 'best.pth')
    else:
        torch.save(state, opt.save_path + '{}epoch.pth'.format(epoch))
