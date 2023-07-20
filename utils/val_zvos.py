import _init_paths

import os
import pickle
import warnings

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.metrics.vos import f_boundary, jaccard

def print_all_keys(data_dict, level: int = 0):
    level += 1
    if isinstance(data_dict, dict):
        for k, v in data_dict.items():
            print(f" {'|=' * level}>> {k}")
            print_all_keys(v, level=level)
    elif isinstance(data_dict, (list, tuple)):
        for item in data_dict:
            print_all_keys(item, level=level)
    else:
        return

def get_mean_recall_decay_for_video(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.

    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values[1:-1] > 0.5)

    # Compute decay as implemented in Matlab
    per_frame_values = per_frame_values[1:-1]  # Remove first frame

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i] : ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


def _read_and_eval_file(mask_video_path: str, pred_video_path: str, frame_name: str):
    # print(frame_name)
    frame_mask_path = os.path.join(mask_video_path, frame_name)
    # print(frame_mask_path)
    frame_pred_path = os.path.join(pred_video_path, frame_name[0:-4]+'.png')
    # print(frame_pred_path)
    frame_mask = cv2.imread(frame_mask_path, 0)  # h, w
    h,w = frame_mask.shape
    # print(h,w)
    frame_pred = cv2.imread(frame_pred_path, 0)
    frame_pred = cv2.resize(frame_pred,(w,h),cv2.INTER_NEAREST)

####
    binary_frame_mask = (frame_mask > 0).astype(np.float32)
    binary_frame_pred = (frame_pred > 128).astype(np.float32)

    # binary_frame_mask = (frame_mask > 127).astype(np.float32)  # object
    # binary_frame_pred = (frame_pred > 127).astype(np.float32)

    J_score = jaccard.db_eval_iou(
        annotation=binary_frame_mask, segmentation=binary_frame_pred
    )
    # print(J_score)
    F_score = f_boundary.db_eval_boundary(
        foreground_mask=binary_frame_pred, gt_mask=binary_frame_mask
    )
    return J_score, F_score


def _eval_video_sequence(
    mask_path: str, gt_path: str, video_name: str, ignore_head: bool, ignore_tail: bool
):
    # print(f"processing {video_name}...")

    mask_video_path = os.path.join(gt_path, video_name, 'GT')
    pred_video_path = os.path.join(mask_path, video_name)

    mask_frame_path_list = sorted(os.listdir(mask_video_path))
    if ignore_head:
        mask_frame_path_list = mask_frame_path_list[1:]
    if ignore_tail:
        mask_frame_path_list = mask_frame_path_list[:-1]
    # for frame_name in mask_frame_path_list:
    #     print(frame_name)
    frame_score_list = [
        _read_and_eval_file(
            mask_video_path=mask_video_path,
            pred_video_path=pred_video_path,
            frame_name=frame_name,
        )
        for frame_name in mask_frame_path_list
    ]
    if ignore_head:
        frame_score_list = [[np.nan, np.nan]] + frame_score_list
    if ignore_tail:
        frame_score_list += [[np.nan, np.nan]]
    frame_score_array = np.asarray(frame_score_list)
    M, O, D = zip(
        *[
            get_mean_recall_decay_for_video(frame_score_array[:, i])
            for i in range(frame_score_array.shape[1])
        ]
    )
    return {
        video_name: {
            "pre_frame": frame_score_array,
            "mean": np.asarray(M),
            "recall": np.asarray(O),
            "decay": np.asarray(D),
        }
    }


def get_method_score_dict(
    epoch: int,
    total_epoch: int,
    test_dataset: str,
    mask_path: str,
    gt_path: str,
    video_name_list: list,
    ignore_head: bool = False,
    ignore_tail: bool = False,
):
    loop = tqdm(video_name_list)
    loop.set_description(f'Epoch [{epoch}/{total_epoch}] ZVOS Eval  [{test_dataset}]')
    video_score_list = Parallel(n_jobs=10)(
        delayed(_eval_video_sequence)(
            mask_path=mask_path,
            gt_path=gt_path,
            video_name=video_name,
            ignore_head=ignore_head,
            ignore_tail=ignore_tail
        )for video_name in loop
    )

    video_score_dict = {
        list(kv.keys())[0]: list(kv.values())[0] for kv in video_score_list
    }
    return video_score_dict


def get_method_average_score_dict(method_score_dict: dict):
    # average_score_dict = {"total": 0, "mean": 0, "recall": 0, "decay": 0}
    average_score_dict = {"Average": {"mean": 0, "recall": 0, "decay": 0}}
    for k, v in method_score_dict.items():
        # average_score_item = np.nanmean(v["pre_frame"], axis=0)
        # average_score_dict[k] = average_score_item
        average_score_dict[k] = {
            "mean": v["mean"],
            "recall": v["recall"],
            "decay": v["decay"],
        }
        # average_score_dict["total"] += average_score_item
        average_score_dict["Average"]["mean"] += v["mean"]
        average_score_dict["Average"]["recall"] += v["recall"]
        average_score_dict["Average"]["decay"] += v["decay"]
    # average_score_dict['Average']["total"] /= len(method_score_dict)
    average_score_dict["Average"]["mean"] /= len(method_score_dict)
    average_score_dict["Average"]["recall"] /= len(method_score_dict)
    average_score_dict["Average"]["decay"] /= len(method_score_dict)
    return average_score_dict

def read_from_file(file_path: str):
    with open(file_path, mode="rb") as f:
        data = pickle.load(f)
    return data

def eval_method_from_data(
    epoch: int, 
    total_epoch: int,
    test_dataset: str, 
    gt_path: str,
    mask_path: str,
    ignore_head=False,
    ignore_tail=False,
):
    eval_video_name_list = os.listdir(mask_path)
    # tervese the each img_flow
    method_score_dict = get_method_score_dict(
        mask_path=mask_path,
        gt_path=gt_path,
        video_name_list=eval_video_name_list,
        ignore_head=ignore_head,
        ignore_tail=ignore_tail,
        test_dataset=test_dataset,
        epoch=epoch,
        total_epoch=total_epoch
    )
    # get the average score
    average_score_dict = get_method_average_score_dict(
        method_score_dict=method_score_dict
    )

    # show the results
    eval_video_name_list += ["Average"]

    J_mean, F_mean = average_score_dict['Average']['mean']
    J_mean, F_mean = round(J_mean, 3), round(F_mean, 3)
    return J_mean, F_mean

def compute_J_F(epoch, total_epoch, test_dataset, test_path, pred_path, ignore_head=False, ignore_tail=False):
    gt_path = os.path.join(test_path, test_dataset)
    mask_path = os.path.join(pred_path, test_dataset)

    J_mean, F_mean = eval_method_from_data(epoch, total_epoch, test_dataset, mask_path=mask_path, gt_path=gt_path)
    return J_mean, F_mean

if __name__ == "__main__":

    epoch = 1
    total_epoch = 1
    test_dataset = 'DAVIS'
    test_path = "../dataset/TestSet"
    pred_path = './test_results/Isomer_Results'
    J_mean, F_mean = compute_J_F(epoch, total_epoch, test_dataset, test_path, pred_path, ignore_tail=True)
    print('J_mean:', J_mean, 'F_mean:', F_mean, pred_path)


