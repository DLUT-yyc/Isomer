import os
import cv2
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Parameters():
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Training Config
        parser.add_argument('--seed', type=int, default=666)
        parser.add_argument("--data_augmentation", type=str2bool, default='True')
        parser.add_argument('--epoch', type=int, default=300)
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--lr', type=float, default=6e-5)
        parser.add_argument('--train_batchsize', type=int, default=8)
        parser.add_argument('--img_size', type=int, default=512)
        parser.add_argument('--optimizer', type=str, default='AdamW')
        parser.add_argument("--ms_train", type=str2bool, default='False')
        parser.add_argument("--save_model", type=str2bool, default='True')
        parser.add_argument('--trainset', type=str)
        parser.add_argument('--save_path', type=str)
        parser.add_argument('--restore_from', type=str)

        parser.add_argument('--val_batchsize', type=int, default=48)
        parser.add_argument('--val_every_epoch', type=int, default=5)

        parser.add_argument("--gpu", type=str, default='0')
        parser.add_argument('--local_rank', type=int)

        parser.add_argument('--encoder', type=str, default='swin_tiny')
        parser.add_argument('--fusion_module_dropout', type=float, default=0.1)
        parser.add_argument('--seghead_dropout', type=float, default=0.1)
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--num_points', type=int, default=121)
        parser.add_argument('--num_blocks', type=int, default=1)
        parser.add_argument('--ffn_dim_ratio', type=int, default=8)
        parser.add_argument('--num_attn_heads', type=int, default=8)

        parser.add_argument('--train_root', type=str, default='../dataset/TrainSet/SampleYTBVOS-30')
        parser.add_argument('--val_root', type=str, default='../dataset/TestSet')
        parser.add_argument('--val_dataset', type=str, default='DAVIS')

        # Inference Config
        parser.add_argument('--infer_dataset', type=str)
        parser.add_argument('--infer_model_path', type=str)
        parser.add_argument('--infer_save', type=str)
        parser.add_argument('--infer_dataset_path', type=str)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        if args.data_augmentation:
            cv2.ocl.setUseOpenCL(False)
            cv2.setNumThreads(0)
        return args
