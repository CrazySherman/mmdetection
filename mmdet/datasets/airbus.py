import os
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, show_ann, random_scale




class AirbusKaggle(Dataset):
	def __init__(self, img_scale, data_root, img_norm_cfg,
				flip_ratio=0,
				size_divisor=None,
				test_mode=False,

				):
		self.data_root = data_root
		#TODO prepare masks from raw encodings or precomputed files?
		self.img_ids = []
		self.masks = dict()  

		# filter images with no ships, this important otherwise get(idx) will return null to dataloader
		if not test_mode:
			self._filter_images()
		# color channel order and normalize configs
        self.img_norm_cfg = img_norm_cfg
		
		self.size_divisor = size_divisor
		# padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
		self.transform =   ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
		self.mask_transform = MaskTransform()
		self.flip_ratio = flip_ratio
		# (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        # set group flag for the sampler 
        self._set_group_flag()

	#TODO add bbox? may or may not be helpful 
	def __getitem__(self, idx):
		"""
			generated transformed image, as well the masks for each ground truth box
			for airbus competition, there's actually no gt boxes, so we'd have to parse the encoding map 
			and obtain boxes from it 
		"""
		if self.test_mode:
			return self.prepare_test_img(idx)

		image_id = self.img_ids[idx]
		masks = self.masks[image_id]
		img = mmcv.imread(os.path.join(self.data_root, image_id))
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


		img_meta = dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
		data = dict(
				img=img,
				img_meta= img_meta,
				gt_bboxes= None,
				gt_labels= None,
				gt_masks= DC(gt_masks, cpu_only=True)
			)
		return data

	def __len__(self):
        return len(self.img_ids)
	
	def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self.img_ids), dtype=np.uint8)
        for i in range(len(self.img_ids)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    #TODO implement this.
    def _filter_images(self):
    	pass

    def prepare_test_img(self, idx):
    	img = mmcv.imread(os.path.join(self.data_root, img_ids[idx]))
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


