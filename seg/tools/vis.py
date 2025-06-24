import sys
sys.path.append(sys.path[0] + "/..")
from mmseg.apis import inference_model, init_model
import torch
import numpy as np
import random
from mmseg.vimm import TopkRouter
import cv2


# python tools/test.py configs/seg/vimms3_vai_test.py segresults/vimms3/iter_10000.pth --tta
def set_drop_router(model, new_drop_prob):
    for module in model.modules():
        if isinstance(module, TopkRouter):
            module.drop_router = new_drop_prob
            print(f"Updated drop_router to {new_drop_prob} for {module}")

def infer(model, img_path):
    result = inference_model(model, img_path)
    mask = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    palette_bgr = {
        0: (255, 0, 0),  # RGB (0, 0, 255) → BGR
        1: (211, 116, 220),  # RGB (220, 116, 211) → BGR
        2: (255, 255, 0),  # RGB (0, 255, 255) → BGR
        3: (0, 255, 0),  # RGB (0, 255, 0) → BGR (相同)
        4: (0, 255, 255),  # RGB (255, 255, 0) → BGR
        5: (0, 0, 255),  # RGB (255, 0, 0) → BGR
        6: (0, 0, 0)  # RGB (0, 0, 0) → BGR (相同)
    }
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in palette_bgr.items():
        rgb_mask[mask == class_id] = color
    cv2.imwrite('vis/vai_4/c60.png', rgb_mask)
    return mask
if __name__ == '__main__':
    seed = 42
    random.seed(seed)  # Python 内置随机模块
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # 当前GPU
    torch.cuda.manual_seed_all(seed)  # 所有GPU（多卡时）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = 'configs/seg/vimmsmall_vai.py'
    checkpoint = 'segresults_new/compare/vimmsmall_vai/del_linear/best_mIoU_epoch_0.pth'
    # checkpoint = 'checkpoint/vaihingen/small/0.6.pth'
    model = init_model(config, checkpoint, device='cuda:0')
    img_path = '/home/jrf/mamba/data/vaihingen/img_dir/train/area30_1422_2048_1934_2560.png'
    set_drop_router(model, 0.)
    f1 = infer(model, img_path)

