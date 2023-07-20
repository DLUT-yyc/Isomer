import _init_paths

import torch
from config import Parameters
opt = Parameters().parse()

from lib.builder import VOSNet

from utils.init import *
from utils.apis import *

def main(opt):

    writer = init_tensorboard(opt)

    init_seed(opt)
    init_device(opt)

    data_loader, sampler, iters_pre_epoch, total_iters = init_train_dataloader(opt)

    if opt.val_dataset != 'None':
        val_data_loader = init_val_dataloader(opt, opt.val_dataset)

    model = VOSNet(opt)

    model = init_ddp_model(model, opt)
    optimizer = init_optimizer(model, opt, data_loader)
    loss_func = init_loss_func(opt)

    start_epoch = 0
    if opt.restore_from != 'None':
        start_epoch, model, optimizer = resume_model_optimizer(model, optimizer, opt, data_loader)

    iters = start_epoch*iters_pre_epoch
    JF_mean_init = 0
    # training
    for epoch in range(start_epoch, opt.epoch):
        epoch += 1
        iters = train_one_epoch(model, loss_func, optimizer, data_loader, sampler, writer, epoch, iters, total_iters, iters_pre_epoch, opt)

        ###### Save And Val ##########
        if epoch % opt.val_every_epoch == 0:
            if (opt.val_dataset != 'None') & (opt.local_rank==0):
                JF_mean = val_one_epoch(model, val_data_loader, writer, epoch, opt, opt.val_dataset)
                if (opt.save_model == True) & (JF_mean > JF_mean_init):
                    save_model_optimizer(model, optimizer, epoch, opt, save_best=True)
                    JF_mean_init = JF_mean
                if opt.save_model == True:
                    save_model_optimizer(model, optimizer, epoch, opt, save_best=False)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main(opt)


