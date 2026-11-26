"""Compare final REEM checkpoints at the original model's negative-image FP budget."""

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_reem_pilot import CONF, SIZE, YOLO, torch
from scripts.summarize_reem_followup import FOLLOWUP, PILOT, SEAMS, matched_gt_indices, read, stratum
from scripts.experiment_identity import atomic_json

MIN_CONF = .05


def predict(weights, targets):
    model = YOLO(str(weights))
    names = list(targets)
    predictions = {}
    for start in range(0, len(names), 16):
        batch = names[start:start + 16]
        files = [str(ROOT / targets[name]["image"]) for name in batch]
        output = model.predict(files, imgsz=SIZE, batch=16, device=0, conf=MIN_CONF, iou=.5,
                               max_det=300, rect=False, verbose=False)
        for name, result in zip(batch, output, strict=True):
            if Path(result.path).stem != name:
                raise ValueError("Prediction path differs from the validation image")
            predictions[name] = (result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())
    return predictions


def measure(predictions, targets, seams, threshold):
    small_hits, all_hits, negative_fp, seam_fp = 0, 0, 0, 0
    for name, (boxes, scores) in predictions.items():
        gt = targets[name]["boxes"]
        if not gt:
            alarm = any(score >= threshold for score in scores)
            negative_fp += alarm
            seam_fp += alarm and name in seams
            continue
        matched, _ = matched_gt_indices(boxes, scores, gt, threshold)
        small_hits += sum(stratum(gt[index]) == "small" for index in matched)
        all_hits += len(matched)
    return dict(conf=threshold, small_tp=small_hits, all_tp=all_hits, negative_fp=negative_fp, seam_fp=seam_fp)


def main():
    torch.set_num_threads(2)
    target_rows = read(ROOT / "reports/iteration_v2/data.json")["rows"]
    targets = {r["name"]: r for r in target_rows if r["split"] == "target_val" and not r["excluded"]}
    seams = {r["name"] for r in read(SEAMS)["rows"] if r["clear_seam"]}
    summary = read(FOLLOWUP / "summary.json")
    experiments = []
    for study in summary["experiments"]:
        folder = PILOT / study["run"]
        output = {}
        for variant in ("original_reem", "directional_reem"):
            predictions = predict(folder / variant / "weights/last.pt", targets)
            fixed = measure(predictions, targets, seams, CONF)
            if any(fixed[key] != study["variants"][variant][key] for key in fixed if key != "conf"):
                raise ValueError(f"Fixed-threshold recalculation differs for {study['seed']}/{variant}")
            output[variant] = fixed
            if variant == "directional_reem":
                directional_predictions = predictions
        budget = output["original_reem"]["negative_fp"]
        # Use the lowest threshold that stays within the original image-level budget.
        negative_scores = sorted((max(scores, default=0.0) for name, (_, scores) in directional_predictions.items()
                                  if not targets[name]["boxes"]), reverse=True)
        threshold = MIN_CONF
        if budget < len(negative_scores):
            threshold = max(threshold, math.nextafter(negative_scores[budget], math.inf))
        output["directional_matched_budget"] = measure(directional_predictions, targets, seams, threshold)
        experiments.append(dict(seed=study["seed"], negative_budget=budget, **output))
        print(experiments[-1], flush=True)
    result = dict(split="target_val", purpose="Exploratory same-validation-set threshold diagnostic",
                  min_prediction_conf=MIN_CONF, fixed_conf=CONF, experiments=experiments)
    atomic_json(FOLLOWUP / "阈值诊断.json", result)


if __name__ == "__main__":
    main()
