"""Freeze the CTCD image/mask pairs and evaluate a single locally verified model."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_v3_sizes import matched_gt_indices, stratum
from scripts.prepare_data import mask_to_boxes
from scripts.run_reem_refinement import OUTPUT, file_sha256, torch, read, predict
from scripts.run_reem_pilot import SIZE
from scripts.experiment_identity import atomic_json

REVISION = "d7f60e61914776221f38208d5064a0246887ef4e"
SOURCE = ROOT / "data/external/ctcd_d7f60e6"
FILES = SOURCE / "Complex Tunnel Crack Dataset (CTCD)"
MANIFEST = OUTPUT / "ctcd_manifest.json"
SOURCE_SIZE = 256


def prepare():
    head = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise ValueError("Unexpected CTCD revision")
    rows, excluded = {}, []
    for split, count in (("train", 250), ("val", 42)):
        images = sorted((FILES / split).glob("*.bmp"))
        masks = sorted((FILES / (split + "annot")).glob("*.bmp"))
        if len(images) != count or [p.stem for p in images] != [p.stem for p in masks]:
            raise ValueError(f"Incorrect image/mask inventory: {split}")
        for image, mask in zip(images, masks, strict=True):
            with Image.open(image) as im, Image.open(mask) as label:
                im.load()
                if im.size != (SOURCE_SIZE, SOURCE_SIZE) or label.size != im.size:
                    raise ValueError(f"Unexpected CTCD image or mask size: {image}")
                annotation = np.asarray(label.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST))
            boxes, _ = mask_to_boxes(annotation)
            if not boxes:
                excluded.append(image.relative_to(ROOT).as_posix())
                continue
            if image.stem in rows:
                raise ValueError(f"Duplicate image ID: {image}")
            rows[image.stem] = dict(split=split, image=image.relative_to(ROOT).as_posix(),
                                    mask=mask.relative_to(ROOT).as_posix(), boxes=boxes,
                                    image_sha256=file_sha256(image), mask_sha256=file_sha256(mask))
    payload = dict(revision=REVISION, license="CC BY 4.0", inventory_images=292, evaluated_images=len(rows),
                   conversion="Nearest-neighbor 2x mask resize; shared mask_to_boxes at 512 px; exclude tile without derived box.",
                   excluded=excluded, rows=rows)
    if MANIFEST.exists():
        if read(MANIFEST) != payload:
            raise ValueError("Already frozen CTCD manifest differs")
    else:
        atomic_json(MANIFEST, payload)
    print("CTCD", len(rows), "boxed images;", len(excluded), "excluded", flush=True)
    return payload


def measure_external(predictions, targets, confidence):
    totals = dict(tp=0, fp=0, fn=0, small_tp=0, small_gt=0)
    for name, (boxes, scores) in predictions.items():
        scale = SIZE / SOURCE_SIZE
        scaled = [[coordinate * scale for coordinate in box] for box in boxes]
        gt = targets[name]["boxes"]
        matched, (tp, fp, fn) = matched_gt_indices(scaled, scores, gt, confidence)
        for key, count in (("tp", tp), ("fp", fp), ("fn", fn),
                           ("small_tp", sum(stratum(gt[idx]) == "small" for idx in matched)),
                           ("small_gt", sum(stratum(box) == "small" for box in gt))):
            totals[key] += count
    return dict(confidence=confidence, **totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation", nargs="?", type=Path)
    args = parser.parse_args()
    dataset = prepare()
    if args.confirmation is None:
        return
    path = args.confirmation.resolve()
    if path.name != "comparison.json" or not path.is_relative_to(OUTPUT.resolve()):
        parser.error("Expected a local confirmation comparison.json")
    verified = read(path.with_name("test_verification.json"))
    if not verified["local_verification_passed"] or verified["confirmation_sha256"] != file_sha256(path):
        raise ValueError("External evaluation requires a confirmed candidate and completed local test gate")
    output = path.with_name("ctcd_verification.json")
    if output.exists():
        raise ValueError("Preserve the existing external evaluation")

    torch.set_num_threads(2)
    models = read(path)["experiments"]
    targets = dataset["rows"]
    result = dict(status="running", revision=REVISION, dataset_sha256=file_sha256(MANIFEST),
                  images=len(targets), negative_images=0, experiments={})
    atomic_json(output, result)
    for seed, variants in models.items():
        result["experiments"][seed] = {}
        for variant, model in variants.items():
            weights = ROOT / model["checkpoint"]
            if file_sha256(weights) != model["weights_sha256"]:
                raise ValueError(f"Checkpoint altered: {weights}")
            metrics = measure_external(predict(weights, targets), targets, model["matched"]["conf"])
            result["experiments"][seed][variant] = metrics
            print(seed, variant, metrics, flush=True)
        atomic_json(output, result)
    result["status"] = "complete"
    atomic_json(output, result)


if __name__ == "__main__":
    main()
