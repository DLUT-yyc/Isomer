import _init_paths

import time
import torch
import numpy as np
from tqdm import tqdm

class Eval_thread():
    def __init__(self, loader, method, dataset):
        self.loader = loader
        self.method = method
        self.dataset = dataset

    def run(self, epoch=1, total_epoch=1):
        # print('eval: {} dataset with {} method.'.format(self.dataset, self.method))
        start_time = time.time()
        beta2 = 0.3
        alpha = 0.5
        mae_dict = dict()
        F_dict = dict()
        E_dict = dict()
        S_dict = dict()
        with torch.no_grad():
            loop = tqdm(self.loader)
            for v_name, preds, gts in loop:
                loop.set_description(f'Epoch [{epoch}/{total_epoch}] VSOD Eval  [{self.dataset}]')
                preds = preds.cuda()
                gts = gts.cuda()

                ####### MAE ######
                mean = torch.abs(preds - gts).mean()
                assert mean == mean, "mean is NaN"  # for Nan
                mae_dict[v_name] = mean

                # F Measure Score
                f_score = 0
                # E Measure Score
                e_score = torch.zeros(256).cuda()
                # S Measure Score
                sum_Q = 0
                for pred, gt in zip(preds, gts):
                    # F-Measure
                    prec, recall = self._eval_pr(pred, gt, 256)
                    f_score += (1 + beta2) * prec * recall / (beta2 * prec + recall+1e-10)
                    assert (f_score == f_score).all()  # for Nan
                    # E-Measure
                    e_score += self._eval_e(pred, gt, 256)
                    # S-Measure
                    y = gt.mean()
                    # if y < 1e-4: print('!!!!!!!!', y)
                    if y < 1e-4:
                        x = pred.mean()
                        Q = 1.0 - x
                    elif y == 1:
                        x = pred.mean()
                        Q = x
                    else:
                        gt[gt >= 0.5] = 1
                        gt[gt < 0.5] = 0
                        Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                        if Q.item() < 0:
                            Q = torch.FloatTensor([0.0])[0].cuda()
                    assert Q==Q,'Q is NaN'
                    sum_Q += Q

                # F-Measure
                f_score /= len(preds)
                F_dict[v_name] = f_score
                # E-Measure
                e_score /= len(preds)
                E_dict[v_name] = e_score
                # S-Measure
                S_video = sum_Q / len(preds)
                S_dict[v_name] = S_video
            # MAE
            MAE_videos_max = torch.mean(torch.tensor(list(mae_dict.values()))).item()
            # Max F-Measure
            F_videos = torch.stack(list(F_dict.values())).mean(dim=0)
            F_videos_max = F_videos.max().item()
            # Max E-Measure
            E_videos = torch.stack(list(E_dict.values())).mean(dim=0)
            E_videos_max = E_videos.max().item()
            # S-Measure
            S_videos_mean = torch.mean(torch.tensor(list(S_dict.values()))).item()

            return [MAE_videos_max, F_videos_max, E_videos_max, S_videos_mean], \
                    'Epoch [{}/{}] Validation [{}]:  MAE: {:.3f}   Max F-Measure: {:.3f}   Max E-Measure: {:.3f}   Max S-Measure: {:.3f}'.format( \
                    epoch, total_epoch, self.dataset, MAE_videos_max, F_videos_max, E_videos_max, S_videos_mean)

    def _eval_e(self, y_pred, y, num):
        h, w = y.shape
        pred = y_pred.expand(num, h, w)
        gt = y.expand(num, h, w)
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda().reshape(num, 1)
        mask = thlist.expand(num, h*w).reshape(num, h, w)
        pred_threshold = torch.where(pred >= mask, 1, 0).float()
        fm = pred_threshold - torch.mean(pred_threshold, dim=(1,2)).reshape(num, 1).expand(num, h*w).reshape(num, h, w)
        gt = gt - torch.mean(gt, dim=(1,2)).reshape(num, 1).expand(num, h*w).reshape(num, h, w)
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score = torch.sum(enhanced, dim=(1,2)) / (y.numel() - 1 + 1e-20)
        return score

    def _eval_pr(self, y_pred, y, num):
        h, w = y.shape
        pred = y_pred.expand(num, h, w)
        gt = y.expand(num, h, w)
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda().reshape(num, 1)
        mask = thlist.expand(num, h*w).reshape(num, h, w)
        pred_threshold = torch.where(pred >= mask, 1, 0).float()
        tp = torch.sum(pred_threshold * gt, dim=(1,2))
        prec, recall = tp / (torch.sum(pred_threshold, dim=(1,2)) + 1e-20), tp / (torch.sum(gt, dim=(1,2)) + 1e-20)
        return prec, recall

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            total = gt.sum()
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

