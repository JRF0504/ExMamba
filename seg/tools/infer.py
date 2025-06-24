# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append(sys.path[0] + "/..")
import argparse
import os
import torch
import os.path as osp
from copy import deepcopy
from mmseg.apis import inference_model, init_model
from mmengine.config import Config, ConfigDict, DictAction
import numpy as np
from PIL import Image
import mmcv
from scipy import stats
from mmengine.dataset import Compose
import cv2
from collections import defaultdict
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPreTrain test (and eval) a model')
    parser.add_argument('config', nargs='?',default='/home/jrf/mamba/src/configs/seg/vimmsmall_loveda_512.py',help='test config file path')
    parser.add_argument('checkpoint',nargs='?',default='/home/jrf/mamba/src/checkpoint/loveda/0_51.08.pth', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='the file to output results.')
    parser.add_argument(
        '--out-item',
        choices=['metrics', 'pred'],
        help='To output whether metrics or predictions. '
        'Defaults to output predictions.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision test')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to enable the Test-Time-Aug (TTA). If the config file '
        'has `tta_pipeline` and `tta_model` fields, use them to determine the '
        'TTA transforms and how to merge the TTA results. Otherwise, use flip '
        'TTA by averaging classification score.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # enable automatic-mixed-precision test
    if args.amp:
        cfg.test_cfg.fp16 = True

    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- TTA related args --------------------
    if args.tta:
        if 'tta_model' not in cfg:
            cfg.tta_model = dict(type='mmpretrain.AverageClsScoreTTA')
        if 'tta_pipeline' not in cfg:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            cfg.tta_pipeline = deepcopy(test_pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [test_pipeline[-1]],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # ----------------- Default dataloader args -----------------
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        collate_fn=dict(type='default_collate'),
    )

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False

    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg

def fast_mode(stacked):
    """
    stacked: (N, H, W) → 多数投票 → (H, W)
    """
    N, H, W = stacked.shape
    flattened = stacked.reshape(N, -1)  # (N, H*W)
    result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=flattened).astype(np.uint8)
    return result.reshape(H, W)

def tta_inference_from_model(model, img_path, scales=(0.75, 1.0, 1.25), use_flip=True):
    test_pipeline_cfg = model.cfg.test_pipeline.copy()
    if test_pipeline_cfg[0]['type'] == 'LoadImageFromFile':
        test_pipeline_cfg[0]['type'] = 'LoadImageFromNDArray'
    for t in test_pipeline_cfg:
        if t.get('type') == 'LoadAnnotations':
            test_pipeline_cfg.remove(t)
    pipeline = Compose(test_pipeline_cfg)
    img_ori = mmcv.imread(img_path)
    H, W = img_ori.shape[:2]
    all_preds = []

    for scale in scales:
        img_scaled = mmcv.imresize(img_ori, (int(W * scale), int(H * scale)))
        pred = _infer_single(img_scaled, pipeline, model, ori_shape=(H, W))
        all_preds.append(pred)

        if use_flip:
            img_flipped = mmcv.imflip(img_scaled, direction='horizontal')
            pred_flip = _infer_single(img_flipped, pipeline, model, ori_shape=(H, W))
            pred_flip = np.fliplr(pred_flip)
            all_preds.append(pred_flip)

    stacked = np.stack(all_preds, axis=0).astype(np.uint8)
    final_mask  = fast_mode(stacked)
    return final_mask


def _infer_single(img, pipeline, model, ori_shape):

    data = defaultdict(list)
    data_ = dict(img=img)
    data_ = pipeline(data_)
    data['inputs'].append(data_['inputs'])
    data['data_samples'].append(data_['data_samples'])
    with torch.no_grad():
        result = model.test_step(data)
    pred = result[0].pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred

def main():
    args = parse_args()

    if args.out is None and args.out_item is not None:
        raise ValueError('Please use `--out` argument to specify the '
                         'path of the output file before using `--out-item`.')

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)


    model = init_model(args.config, args.checkpoint, device='cuda:0')
    img_path = '/home/jrf/mamba/data/loveDA/img_dir/test'
    out_path = 'loveda_pred_4w'
    os.makedirs(out_path, exist_ok=True)
    file_list = os.listdir(img_path)

    for file in tqdm(file_list, desc="Processing images"):
        full_img_path = os.path.join(img_path, file)

        if args.tta:
            seg_array = tta_inference_from_model(model, full_img_path)
        else:
            result = inference_model(model, full_img_path)
            seg_array = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)


        img = Image.fromarray(seg_array)
        img.save(os.path.join(out_path, file))

if __name__ == '__main__':
    main()
