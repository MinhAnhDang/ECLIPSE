import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

BTCV_SEM_SEG_CATEGORIES = ["background","spleen","rkid","lkid","gall","eso","liver","sto","aorta","IVC","veins","pancreas","rad","lad"]

def register_btcv_sem(root):
    root = os.path.join(root, "BTCV")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations", dirname)
        name = f"my_btcv_sem_seg_{name}"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg"))
        MetadataCatalog.get(name).set(
            stuff_classes=BTCV_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_btcv_sem(_root)