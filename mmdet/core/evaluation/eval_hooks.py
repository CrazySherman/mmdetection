import os
import os.path as osp
import shutil
import time
from itertools import chain

import mmcv
import numpy as np
import torch
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

from torch.utils.data import Dataset

from .coco_utils import results2json, fast_eval_recall
from mmdet import datasets


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.lock_dir = None

    def _barrier(self, rank, world_size):
        """Due to some issues with `torch.distributed.barrier()`, we have to
        implement this ugly barrier function.
        """
        if rank == 0:
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                while not (osp.exists(tmp)):
                    time.sleep(1)
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                os.remove(tmp)
        else:
            tmp = osp.join(self.lock_dir, '{}.pkl'.format(rank))
            mmcv.dump([], tmp)
            while osp.exists(tmp):
                time.sleep(1)

    def before_run(self, runner):
        self.lock_dir = osp.join(runner.work_dir, '.lock_map_hook')
        if runner.rank == 0:
            if osp.exists(self.lock_dir):
                shutil.rmtree(self.lock_dir)
            mmcv.mkdir_or_exist(self.lock_dir)

    def after_run(self, runner):
        if runner.rank == 0:
            shutil.rmtree(self.lock_dir)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            for _ in range(batch_size):
                prog_bar.update()

        if runner.rank == 0:
            print('\n')
            self._barrier(runner.rank, runner.world_size)
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            self._barrier(runner.rank, runner.world_size)
        self._barrier(runner.rank, runner.world_size)

    def evaluate(self):
        raise NotImplementedError


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0.json')
        results2json(self.dataset, results, tmp_file)

        res_types = ['bbox',
                     'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        cocoDt = cocoGt.loadRes(tmp_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            field = '{}_mAP'.format(res_type)
            runner.log_buffer.output[field] = cocoEval.stats[0]
        runner.log_buffer.ready = True
        os.remove(tmp_file)



# refer to the notebook airbusdataset.ipynb
def IoU(pred, targs):
    pred = (pred > 0.5).astype(float)
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)

def get_score(pred, true):
    n_th = 10
    b = 4
    thresholds = [0.5 + 0.05*i for i in range(n_th)]
    n_masks = len(true)
    n_pred = len(pred)
    # if any of the masks are 0, it's either 0/1 score
    if n_pred == 0 and n_masks == 0:
        return 1
    elif n_pred == 0 or n_masks == 0:
        return 0
    ious = []
    score = 0
    for mask in true:
        buf = []
        for p in pred: buf.append(IoU(p,mask))
        ious.append(buf)
    for t in thresholds:   
        tp, fp, fn = 0, 0, 0
        for i in range(n_masks):
            match = False
            for j in range(n_pred):
                if ious[i][j] > t: match = True
            if not match: fn += 1
        
        for j in range(n_pred):
            match = False
            for i in range(n_masks):
                if ious[i][j] > t: match = True
            if match: tp += 1
            else: fp += 1
        score += ((b+1)*tp)/((b+1)*tp + b*fn + fp)       
    return score/n_th

class AirbusEvalF2ScoreHook(Hook):
    """
        Eval hook used by airbus dataset as validation indicator during training
    """
    def __init__(self, dataset, interval=1000):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'val_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_iter(self, runner):
        """
            after <interval> iterations, run eval
        """
        if self.every_n_inner_iters(runner, self.interval):
            runner.logger.info('starting evaluation')
            runner.model.eval()
            scores = []
            prog_bar = mmcv.ProgressBar(len(self.dataset))
            num_empty_ships = 0
            num_pred_empty_ships = 0
            for idx in range(len(self.dataset)):
                data, gt_masks = self.dataset[idx]
                with torch.no_grad():
                    result = runner.model(return_loss=False, rescale=True, **data)
                assert type(gt_masks) == list
                # evaluate on each image in batch
                # assert batch size of input - output are aligned 
                assert len(gt_masks) == len(result[1]), '{} vs {}'.format(len(gt_masks), len(result[1]))
                for i, mask_res in enumerate(result[1]):
                    if len(gt_masks[i]) == 0:
                        num_empty_ships += 1
                    # result structure:
                    pred_masks_rle = mask_res
                    if len(pred_masks_rle) == 0:
                        num_pred_empty_ships += 1

                    pred_masks = [mask_util.decode(m) for m in pred_masks_rle]
                    scores.append(get_score(pred_masks, gt_masks[i]))
                # cur batch has finished
                prog_bar.update()

            runner.logger.info('finshed evaluation on {} number of images; empty ships: {}; pred empty ships: {}'.format(len(scores), num_empty_ships, num_pred_empty_ships))
            runner.logger.info('evaluation score: {}'.format(np.array(scores).mean()))

