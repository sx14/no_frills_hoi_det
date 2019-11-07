import os

from utils.argparse_utils import manage_required_args
from exp.experimenter import *
from data.hico.hico_constants import HicoConstants
from utils.constants import Constants, ExpConstants
import exp.hoi_classifier.data.hoi_candidates as hoi_candidates
import exp.hoi_classifier.data.label_hoi_candidates as label_hoi_candidates
from exp.hoi_classifier.models.hoi_classifier_model import HoiClassifierConstants
import exp.hoi_classifier.train as train
import exp.hoi_classifier.eval as evaluate
from exp.hoi_classifier.data.features_dataset import FeatureConstants
import exp.hoi_classifier.data.cache_box_features as cache_box_features
import exp.hoi_classifier.data.cache_pose_features as cache_pose_features
import exp.hoi_classifier.data.assign_pose_to_human_candidates as \
    assign_pose_to_human_candidates
import exp.hoi_classifier.vis.top_boxes_per_hoi as \
    vis_top_boxes_per_hoi


parser.add_argument(
    '--gen_hoi_cand',
    default=False,
    action='store_true',
    help='Apply this flag to generate hoi candidates')
parser.add_argument(
    '--label_hoi_cand',
    default=False,
    action='store_true',
    help='Apply this flag to label hoi candidates')
parser.add_argument(
    '--subset',
    type=str,
    choices=['train','train_val','val','test'],
    help='Apply this flag to specify subset of data')
parser.add_argument(
    '--imgs_per_batch',
    type=int,
    default=1,
    help='Number of images per batch')
parser.add_argument(
    '--fp_to_tp_ratio',
    type=int,
    default=1000,
    help='Number of images per batch')
parser.add_argument(
    '--model_num',
    type=int,
    help='Specify model number to evaluate')
parser.add_argument(
    '--verb_given_appearance',
    default=False,
    action='store_true',
    help='Use verb_given_human/object_appearance factor')
parser.add_argument(
    '--verb_given_human_appearance',
    default=False,
    action='store_true',
    help='Set verb_given_human_appearance factor')
parser.add_argument(
    '--verb_given_object_appearance',
    default=False,
    action='store_true',
    help='Set verb_given_object_appearance factor')
parser.add_argument(
    '--verb_given_boxes_and_object_label',
    default=False,
    action='store_true',
    help='Use verb_given_boxes_and_object_label factor')
parser.add_argument(
    '--verb_given_human_pose',
    default=False,
    action='store_true',
    help='Use verb_given_human_pose factor')
parser.add_argument(
    '--rcnn_det_prob',
    default=False,
    action='store_true',
    help='Use detection prob from Faster-RCNN')


def exp_gen_and_label_hoi_cand():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'],
        optional_args=['gen_hoi_cand','label_hoi_cand'])
    if len(not_specified_args) > 0:
        return
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = args.subset

    data_const = HicoConstants()
    data_const.selected_dets_hdf5 = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/select_confident_boxes_in_hico/' + \
        'selected_coco_cls_dets.hdf5')

    if args.gen_hoi_cand:
        # 全组合human和object，不考虑是否命中
        print('Generating HOI candidates from Faster-RCNN dets...')
        hoi_candidates.generate(exp_const,data_const)
    
    if args.label_hoi_cand:
        # 给上一步全组合产生的candidates标注是否命中gt
        print('Labelling HOI candidates from Faster-RCNN dets...')
        data_const.hoi_cand_hdf5 = os.path.join(
            exp_const.exp_dir,
            f'hoi_candidates_{exp_const.subset}.hdf5')
        label_hoi_candidates.assign(exp_const,data_const)


def exp_cache_box_feats():
    # 产生相对位置特征
    args = parser.parse_args()
    
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'])
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = args.subset

    data_const = HicoConstants()
    data_const.hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'hoi_candidates_{exp_const.subset}.hdf5')

    cache_box_features.main(exp_const,data_const)


def exp_assign_pose_to_human_cand():
    # 为candidates中的所有human分配pose skeleton
    args = parser.parse_args()
    
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'])
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = args.subset

    data_const = HicoConstants()
    data_const.hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'hoi_candidates_{exp_const.subset}.hdf5')
    data_const.human_pose_dir = os.path.join(
        data_const.proc_dir,
        'human_pose')
    data_const.num_keypoints = 18

    assign_pose_to_human_candidates.main(exp_const,data_const)


def exp_cache_pose_feats():

    args = parser.parse_args()
    
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['subset'])
    
    exp_name = 'hoi_candidates'
    exp_const = ExpConstants(exp_name=exp_name)
    exp_const.subset = args.subset

    data_const = HicoConstants()
    data_const.hoi_cand_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'hoi_candidates_{exp_const.subset}.hdf5')
    data_const.human_cands_pose_hdf5 = os.path.join(
        exp_const.exp_dir,
        f'human_candidates_pose_{exp_const.subset}.hdf5')
    data_const.num_keypoints = 18

    cache_pose_features.main(exp_const,data_const)


def exp_train():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['imgs_per_batch','fp_to_tp_ratio'],
        optional_args=[
            'verb_given_appearance',                # check
            'verb_given_human_appearance',
            'verb_given_object_appearance',
            'verb_given_boxes_and_object_label',    # check
            'verb_given_human_pose',                # check
            'rcnn_det_prob'])                       # check

    exp_name = 'factors'
    if args.rcnn_det_prob:
        exp_name += '_rcnn_det_prob'
    if args.verb_given_appearance:
        exp_name += '_appearance'
    if args.verb_given_human_appearance:
        exp_name += '_human_appearance'
    if args.verb_given_object_appearance:
        exp_name += '_object_appearance'
    if args.verb_given_boxes_and_object_label:
        exp_name += '_boxes_and_object_label'
    if args.verb_given_human_pose:
        exp_name += '_human_pose'
    
    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_classifier')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.log_dir = os.path.join(exp_const.exp_dir,'log')
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_epochs = 10
    exp_const.imgs_per_batch = args.imgs_per_batch
    exp_const.lr = 1e-3

    # 数据集参数，主要是路径
    data_const_train = FeatureConstants(subset='train')
    data_const_val = FeatureConstants(subset='val')

    # 各模块参数
    model_const = Constants()
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_human_appearance = args.verb_given_human_appearance
    model_const.hoi_classifier.verb_given_object_appearance = args.verb_given_object_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.verb_given_human_pose = args.verb_given_human_pose
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob

    train.main(exp_const,data_const_train,data_const_val,model_const)


def exp_eval():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['model_num'],
        optional_args=[
            'verb_given_appearance',
            'verb_given_human_appearance',
            'verb_given_object_appearance',
            'verb_given_boxes_and_object_label',
            'verb_given_human_pose',
            'rcnn_det_prob'])

    exp_name = 'factors'
    if args.rcnn_det_prob:
        exp_name += '_rcnn_det_prob'
    if args.verb_given_appearance:
        exp_name += '_appearance'
    if args.verb_given_human_appearance:
        exp_name += '_human_appearance'
    if args.verb_given_object_appearance:
        exp_name += '_object_appearance'
    if args.verb_given_boxes_and_object_label:
        exp_name += '_boxes_and_object_label'
    if args.verb_given_human_pose:
        exp_name += '_human_pose'

    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_classifier')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')

    data_const = FeatureConstants(subset='test')
    data_const.balanced_sampling = False
    
    model_const = Constants()
    model_const.model_num = args.model_num
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_human_appearance = args.verb_given_human_appearance
    model_const.hoi_classifier.verb_given_object_appearance = args.verb_given_object_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.verb_given_human_pose = args.verb_given_human_pose
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob
    model_const.hoi_classifier.model_pth = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')
    evaluate.main(exp_const,data_const,model_const)


def exp_top_boxes_per_hoi():
    args = parser.parse_args()
    not_specified_args = manage_required_args(
        args,
        parser,
        required_args=['model_num'],
        optional_args=[
            'verb_given_appearance',
            'verb_given_human_appearance',
            'verb_given_object_appearance',
            'verb_given_boxes_and_object_label',
            'verb_given_human_pose',
            'rcnn_det_prob'])

    exp_name = 'factors'
    if args.rcnn_det_prob:
        exp_name += '_rcnn_det_prob'
    if args.verb_given_appearance:
        exp_name += '_appearance'
    if args.verb_given_human_appearance:
        exp_name += '_human_appearance'
    if args.verb_given_object_appearance:
        exp_name += '_object_appearance'
    if args.verb_given_boxes_and_object_label:
        exp_name += '_boxes_and_object_label'
    if args.verb_given_human_pose:
        exp_name += '_human_pose'
     
    out_base_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_classifier')
    exp_const = ExpConstants(
        exp_name=exp_name,
        out_base_dir=out_base_dir)
    exp_const.model_dir = os.path.join(exp_const.exp_dir,'models')
    exp_const.num_to_vis = 10

    data_const = FeatureConstants(subset='test')
    data_const.pred_hoi_dets_h5py = os.path.join(
        exp_const.exp_dir,
        f'pred_hoi_dets_test_{args.model_num}.hdf5')
    hoi_cand_dir = os.path.join(
        os.getcwd(),
        'data_symlinks/hico_exp/hoi_candidates')
    data_const.human_pose_feats_hdf5 = os.path.join(
        hoi_cand_dir,
        'human_pose_feats_test.hdf5')
    data_const.num_pose_keypoints = 18
    
    model_const = Constants()
    model_const.model_num = args.model_num
    model_const.hoi_classifier = HoiClassifierConstants()
    model_const.hoi_classifier.verb_given_appearance = args.verb_given_appearance
    model_const.hoi_classifier.verb_given_boxes_and_object_label = args.verb_given_boxes_and_object_label
    model_const.hoi_classifier.verb_given_human_pose = args.verb_given_human_pose
    model_const.hoi_classifier.rcnn_det_prob = args.rcnn_det_prob
    model_const.hoi_classifier.model_pth = os.path.join(
        exp_const.model_dir,
        f'hoi_classifier_{model_const.model_num}')

    vis_top_boxes_per_hoi.main(exp_const,data_const,model_const)


if __name__=='__main__':
    list_exps(globals())