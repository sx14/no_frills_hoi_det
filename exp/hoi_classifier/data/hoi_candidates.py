import os
import numpy as np
from tqdm import tqdm
import threading
import h5py

import utils.io as io
from utils.constants import save_constants
from data.coco_classes import COCO_CLASSES


class HoiCandidatesGenerator():
    def __init__(self,data_const):
        self.data_const = data_const
        self.hoi_classes = self.get_hoi_classes()
        
    def get_hoi_classes(self):
        hoi_list = io.load_json_object(self.data_const.hoi_list_json)
        hoi_classes = {hoi['id']:hoi for hoi in hoi_list}
        return hoi_classes

    def predict(self,selected_dets):
        pred_hoi_dets = []
        start_end_ids = np.zeros([len(self.hoi_classes),2],dtype=np.int32)
        start_id = 0
        for hoi_id, hoi_info in self.hoi_classes.items():
            # 对于每一个HOI类别
            # 全组合当前图片的所有human和当前HOI类别的object
            # 注意：这里不考虑是否命中，只依据类别是否一致
            dets = self.predict_hoi(selected_dets,hoi_info)
            pred_hoi_dets.append(dets)
            hoi_idx = int(hoi_id)-1
            start_end_ids[hoi_idx,:] = [start_id,start_id+dets.shape[0]]
            start_id += dets.shape[0]
        pred_hoi_dets = np.concatenate(pred_hoi_dets)
        return pred_hoi_dets, start_end_ids

    def predict_hoi(self,selected_dets,hoi_info):
        # 当前图片所有human
        # 当前图片所有hoi对应的object
        # 直接全组合
        hoi_object = ' '.join(hoi_info['object'].split('_'))
        human_boxes = selected_dets['boxes']['person']
        human_scores = selected_dets['scores']['person']
        human_rpn_ids = selected_dets['rpn_ids']['person']
        object_boxes = selected_dets['boxes'][hoi_object]
        object_scores = selected_dets['scores'][hoi_object]
        object_rpn_ids = selected_dets['rpn_ids'][hoi_object]
        num_hoi_dets = human_boxes.shape[0]*object_boxes.shape[0]
        hoi_dets = np.zeros([num_hoi_dets,13])
        hoi_idx = int(hoi_info['id'])-1
        hoi_dets[:,-1] = hoi_idx
        count = 0
        for i in range(human_boxes.shape[0]):
            for j in range(object_boxes.shape[0]):
                hoi_dets[count,:4] = human_boxes[i]
                hoi_dets[count,4:8] = object_boxes[j]
                hoi_dets[count,8:12] = [human_scores[i],object_scores[j], \
                    human_rpn_ids[i],object_rpn_ids[j]]
                count += 1
        return hoi_dets


def generate(exp_const,data_const):
    print(f'Creating exp_dir: {exp_const.exp_dir}')
    io.mkdir_if_not_exists(exp_const.exp_dir)

    save_constants({'exp': exp_const,'data': data_const},exp_const.exp_dir)

    print(f'Reading split_ids.json ...')
    split_ids = io.load_json_object(data_const.split_ids_json)

    print('Creating an object-detector-only HOI detector ...')
    hoi_cand_gen = HoiCandidatesGenerator(data_const)    

    print(f'Creating a hoi_candidates_{exp_const.subset}.hdf5 file ...')
    hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir,f'hoi_candidates_{exp_const.subset}.hdf5')
    f = h5py.File(hoi_cand_hdf5,'w')

    print('Reading selected dets from hdf5 file ...')
    all_selected_dets = h5py.File(data_const.selected_dets_hdf5,'r')

    for global_id in tqdm(split_ids[exp_const.subset]):
        selected_dets = {
            'boxes': {},
            'scores': {},
            'rpn_ids': {}
        }
        start_end_ids = all_selected_dets[global_id]['start_end_ids'].value
        boxes_scores_rpn_ids = \
            all_selected_dets[global_id]['boxes_scores_rpn_ids'].value

        for cls_ind, cls_name in enumerate(COCO_CLASSES):
            start_id,end_id = start_end_ids[cls_ind]
            boxes = boxes_scores_rpn_ids[start_id:end_id,:4]
            scores = boxes_scores_rpn_ids[start_id:end_id,4]
            rpn_ids = boxes_scores_rpn_ids[start_id:end_id,5]
            selected_dets['boxes'][cls_name] = boxes
            selected_dets['scores'][cls_name] = scores
            selected_dets['rpn_ids'][cls_name] = rpn_ids

        pred_dets, start_end_ids = hoi_cand_gen.predict(selected_dets)
        f.create_group(global_id)
        f[global_id].create_dataset(
            'boxes_scores_rpn_ids_hoi_idx',data=pred_dets)
        f[global_id].create_dataset('start_end_ids',data=start_end_ids)

    f.close()