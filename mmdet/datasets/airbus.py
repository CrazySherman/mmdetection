import os
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import logging

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, show_ann, random_scale
import pandas as pd 
from skimage.measure import label, regionprops


# https://www.kaggle.com/inversion/run-length-decoding-quick-start
# see visualization: https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualization
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

# implementation of box generation algorithm
# search cannot be done because of max depth recursion reached
# this algorithm has bugs, think about it
def box_genV1(grid):
    """
        input: grid - ndarray of mask shape: N * M
        outupt: boxes array of cors
    """
    grid = grid.astype(np.int)
    min_area = 10
    N,M = grid.shape
    mask = np.zeros((N,M), np.int)
    bboxes = []
    cur_max = 0
    for i in range(N):
        for j in range(M):
            if not grid[i,j]:
                continue
            elif mask[i,j]:
                continue
            else:
                if i - 1 >= 0 and grid[i - 1, j]:
                    mask[i,j] = mask[i - 1, j]
                elif j - 1 >= 0 and grid[i, j - 1]:
                    mask[i,j] = mask[i, j - 1]
                else:
                    # in this case, it's a new area
                    mask[i,j] = cur_max + 1
                    cur_max += 1
                    
    n_boxes = cur_max
    for i in range(n_boxes):
        idx, idy = np.where(mask == i+1)
        xmin, xmax, ymin, ymax = idx.min(), idx.max(), idy.min(), idy.max()
        if (xmax - xmin) * (ymax - ymin) > min_area:
            bboxes.append((xmin, xmax, ymin, ymax))
        
    return bboxes

# given the mask, generate boxes and mask per box
def get_boxes_and_masks(input_grid):

    def to_coco_format(bboxes):
        """
            This is actually not coco foramt (x, y, w, h), this format is internal to mmdet (xmin, ymin, xmax, ymax)
            input: skimage regionprops format: (min_row, min_col, max_row, max_col)
            return: mmdet internal format; (xmin, ymin, xmax, ymax)
        """
        return [(b[1], b[0], b[3], b[2]) for b in bboxes]

    def box_genV2(grid):
        """
            input: grid
            return: array of tuples
        """
        lbl = label(grid) 
        props = regionprops(lbl)
        return [prop.bbox for prop in props]

    bboxes = box_genV2(input_grid)
    masks = []
    for b in bboxes:
        m = np.zeros(input_grid.shape)
        m[b[0]:b[2],b[1]:b[3]] = input_grid[b[0]:b[2],b[1]:b[3]]
        masks.append(m)
    return to_coco_format(bboxes), masks 

# TODO peformance benchmarking this shit, plus box generation algo
# if slower than disk load time, write them to files
class MaskLoader:
    def __init__(self, data_root):
         # load rle file from csv
        self.masks_rle = pd.read_csv(os.path.join(data_root, 'train_ship_segmentations_v2.csv'))
        # drop the na
        # self.masks_rle.dropna(inplace=True)

    def __getitem__(self, imageId):
        return self._get_mask_from_rle(imageId)

    def _get_mask_from_rle(self, imageId):
        mask = self.masks_rle.loc[self.masks_rle['ImageId'] == imageId, 'EncodedPixels'].dropna().tolist()
        if len(mask) == 0:
            logging.warning('trying to get an empty ship mask')
            return None
        res = np.zeros((768, 768))
        for m in mask:
            res += rle_decode(m)
        return res

    def get_no_ship_image_ids(self):
        null_masks = self.masks_rle['EncodedPixels'].isnull()
        selected = self.masks_rle['ImageId'][null_masks].tolist()
        empty_ships = set()
        for s in selected:
            empty_ships.add(s)
        return empty_ships


class AirbusKaggle(Dataset):
    def __init__(self, img_scale, data_root, img_norm_cfg,
                flip_ratio=0,
                size_divisor=None,
                test_mode=False
                ):
       
        logging.info('starting scanning data root: ', data_root)
        if test_mode:
            self.img_root, self.img_ids = self.get_all_test_image_ids(data_root)
        else:
            self.img_root, self.img_ids = self.get_all_train_image_ids(data_root)

        # MaskLoader as the proxy for panda processing
        # boxes also come from masks
        self.masks =  MaskLoader(data_root) 
        logging.info('Masks successfully loaded')


        # filter images with no ships, this important otherwise get(idx) will return null to dataloader
        if not test_mode:
            logging.info('train mode, start filtering empty ship images...')
            self._filter_images()
        else:
            logging.info('this is test mode')
        self.test_mode = test_mode
        # color channel order and normalize configs
        self.img_norm_cfg = img_norm_cfg
        
        self.size_divisor = size_divisor
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.transform =   ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.mask_transform = MaskTransform()
        self.bbox_transform = BboxTransform()
        self.flip_ratio = flip_ratio
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        # # set group flag for the sampler 
        # self._set_group_flag()
        
        print('Airbus Dataset Summary \n'
            '\ttest_mode: {} \n'
            '\timg_norm_cfg: {} \n'
            '\tsize_divisor: {} \n'
            '\tflip_ratio: {} \n'
            '\timage_scales: {} \n'
            .format(
                self.test_mode,
                self.img_norm_cfg,
                self.size_divisor,
                self.flip_ratio,
                self.img_scales
                ))


    def __getitem__(self, idx):
        """
            generated transformed image, as well the masks for each ground truth box
            for airbus competition, there's actually no gt boxes, so we'd have to parse the encoding map 
            and obtain boxes from it 
        """
        if self.test_mode:
            return self.prepare_test_img(idx)

        image_id = self.img_ids[idx]
        mask = self.masks[image_id]
        # generate masks and boxes from a single mask for that image
        bboxes, masks = get_boxes_and_masks(mask)
        gt_bboxes = np.asarray(bboxes)

        assert len(gt_bboxes) > 0, 'this training image has no boxes/masks'


        img = mmcv.imread(os.path.join(self.img_root, image_id))
        ori_shape = img.shape
         # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        img, img_shape, pad_shape, scale_factor = self.transform(
            img, img_scale, flip)
        # generate ground truth masks and it's corresponding label
        # there's no "label", because it's only ship/noship, does not make sense to have background gt boxes 
        gt_masks = self.mask_transform(masks, pad_shape,
                                               scale_factor, flip)

        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                            flip)
        img_meta = dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
        data = dict(
                img=img,
                img_meta= img_meta,
                gt_bboxes=DC(to_tensor(gt_bboxes)),
                gt_labels= None,
                gt_masks= DC(gt_masks, cpu_only=True)
            )
        return data

    def __len__(self):
        return len(self.img_ids)
    
    # def _set_group_flag(self):
 #        """Set flag according to image aspect ratio.

 #        Images with aspect ratio greater than 1 will be set as group 1,
 #        otherwise group 0.
 #        """
 #        self.flag = np.zeros(len(self.img_ids), dtype=np.uint8)
 #        for i in range(len(self.img_ids)):
 #            img_info = self.img_infos[i]
 #            if img_info['width'] / img_info['height'] > 1:
 #                self.flag[i] = 1

    # during training, remove the images without gt
    def _filter_images(self):
        bad_image_ids = self.masks.get_no_ship_image_ids()
        filtered_imgs = [i for i in self.img_ids if i not in bad_image_ids]
        print('Image Count before filtering: ', len(self))        
        self.img_ids = filtered_imgs
        print('Image Count after filtering: ', len(self))

    def prepare_test_img(self, idx):
        img = mmcv.imread(os.path.join(self.img_root, img_ids[idx]))
        ori_shape = img.shape
        def prepare_single(img, scale, flip, proposal=None):
            """
                transform to predefined size and scale
            """
            _img, img_shape, pad_shape, scale_factor = self.transform(
                img, scale, flip)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack([_proposal, score[:, None]
                                       ]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))

        data = dict(img=imgs, img_meta=img_metas)
        return data

    def get_all_train_image_ids(self, data_root):
        imgs = os.listdir(os.path.join(data_root, 'train_v2'))
        print('Train images are like: ', imgs[:10])
        return os.path.join(data_root, 'train_v2'), imgs
        
    def get_all_test_image_ids(self, data_root):
        imgs = os.listdir(os.path.join(data_root, 'test_v2'))        
        print('Test images are like: ', imgs[:10])
        return os.path.join(data_root, 'test_v2'), imgs



