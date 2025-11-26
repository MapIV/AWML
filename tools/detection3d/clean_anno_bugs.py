# tools/detection3d/check_pred_vs_gt.py
import argparse
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import load_prediction, load_gt
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    cfg = config_factory("detection_cvpr_2019")

    print("[INFO] Loading predictions...")
    pred_boxes, _ = load_prediction(
        args.result_path,
        cfg.max_boxes_per_sample,
        DetectionBox,
        verbose=True
    )
    print("[INFO] Loading GT...")
    gt_boxes = load_gt(
        nusc,
        args.eval_split,
        DetectionBox,
        verbose=True
    )

    pred_tokens = set(pred_boxes.sample_tokens)
    gt_tokens = set(gt_boxes.sample_tokens)

    print("===================================")
    print(f"len(pred_tokens): {len(pred_tokens)}")
    print(f"len(gt_tokens)  : {len(gt_tokens)}")
    missing = gt_tokens - pred_tokens
    extra   = pred_tokens - gt_tokens
    print(f"missing_in_pred (GT - PRED): {len(missing)}")
    print(f"extra_in_pred   (PRED - GT): {len(extra)}")
    print("Some missing_in_pred:", list(missing)[:10])
    print("Some extra_in_pred  :", list(extra)[:10])

if __name__ == "__main__":
    main()
