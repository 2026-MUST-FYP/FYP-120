"""Train and score the three P3 REEM pilot variants on the frozen TACK split."""

import argparse
import csv
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "work/FYP-120"
sys.path[:0] = [str(ROOT), str(REPO)]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import load_checkpoint

from scripts.diagnose_v3_sizes import score_payload
from scripts.evaluate_external import file_sha256
from scripts.evaluate_models import match_counts
from scripts.experiment_identity import atomic_json
from scripts.prepare_iteration import verify_frozen
from scripts.run_iteration import BASELINE, OPTIONS, fixed_summary
from scripts.seeded_trainer import SeededDetectionTrainer
from src.reem_p3 import VARIANTS, build_detector

DATA = ROOT / "data/iteration_v2/target_only.yaml"
SEED = 20261126
CONF = 0.25
SIZE = 512
SETTINGS = dict(OPTIONS, epochs=5, imgsz=SIZE, workers=0, amp=False,
                warmup_epochs=1, close_mosaic=0, plots=False)


def preflight(seed):
    source, _ = load_checkpoint(BASELINE)
    detectors = {name: build_detector(source, name, seed).eval() for name in VARIANTS}
    original = detectors["original_reem"].model[-2]
    directional = detectors["directional_reem"].model[-2]
    for key, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, directional.state_dict()[key], rtol=0, atol=0)
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(2, 64, 17, 19, generator=generator)
    with torch.no_grad():
        torch.testing.assert_close(original(features), directional(features), rtol=0, atol=0)
        image = torch.randn(1, 3, 128, 128, generator=generator)
        for detector in detectors.values():
            prediction = detector(image)[0]
            assert prediction.shape == (1, 5, 336) and torch.isfinite(prediction).all()
    directional.train()
    directional(features).square().mean().backward()
    gradient = directional.direction_gate.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    return {name: sum(p.numel() for p in detector.parameters()) for name, detector in detectors.items()}


def train(name, folder, settings, seed):
    class PilotTrainer(SeededDetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return build_detector(weights, name, seed, verbose=verbose)

    trainer = PilotTrainer(overrides=dict(settings, model=str(BASELINE), data=str(DATA),
                                          project=str(folder), name=name, seed=seed))
    trainer.train()
    run = folder / name
    with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    if len(history) != settings["epochs"] or not (run / "weights/last.pt").is_file():
        raise RuntimeError(f"Incomplete training artifacts: {name}")
    return run, {key.strip(): float(value) for key, value in history[-1].items()}


def evaluate(run, target_rows):
    model = YOLO(str(run / "weights/last.pt"))
    rows = []
    names = list(target_rows)
    for start in range(0, len(names), 16):
        batch = [target_rows[key] for key in names[start:start + 16]]
        paths = [str(ROOT / row["image"]) for row in batch]
        predictions = model.predict(paths, imgsz=SIZE, batch=16, device=0, conf=CONF, iou=.5,
                                    max_det=300, rect=False, verbose=False)
        for target, prediction in zip(batch, predictions, strict=True):
            if target["name"] != Path(prediction.path).stem:
                raise RuntimeError("Validation image and prediction order disagree")
            boxes = prediction.boxes.xyxy.cpu().tolist()
            scores = prediction.boxes.conf.cpu().tolist()
            tp, fp, fn = match_counts(boxes, scores, target["boxes"], CONF)
            rows.append(dict(name=target["name"], image=target["image"], gt=len(target["boxes"]),
                             tp=tp, fp=fp, fn=fn,
                             predictions=[dict(box=b, confidence=s) for b, s in zip(boxes, scores, strict=True)]))
    metrics = fixed_summary(rows)
    metrics["by_box_size"] = score_payload({"rows": rows}, target_rows, CONF, metrics)["groups"]
    atomic_json(run / "validation_predictions.json", {"conf": CONF, "metrics": metrics, "rows": rows})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only verify the models and frozen data.")
    parser.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.seed < 2**32:
        parser.error("epochs must be positive and seed must be an unsigned 32-bit integer")
    torch.set_num_threads(2)
    data = verify_frozen()
    target_rows = {row["name"]: row for row in data["rows"]
                   if row["split"] == "target_val" and not row["excluded"]}
    assert len(target_rows) == data["counts"]["target_val"]["images"] == 426
    assert torch.cuda.is_available(), "This pilot requires the local GPU."
    pretrained_hash = file_sha256(BASELINE)
    if pretrained_hash != "041bc955c3f6cfb7a597d456b07e235cbc0807e4e33448beb5a7a5918fb37576":
        raise RuntimeError("The archived YOLO11n initialization differs from the recorded source.")
    sizes = preflight(args.seed)
    print(f"PREFLIGHT OK: {sizes}, {len(target_rows)} validation images", flush=True)
    if args.check:
        return

    folder = ROOT / "runs/reem_pilot" / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    settings = dict(SETTINGS, epochs=args.epochs)
    metadata = dict(status="running", data_fingerprint=data["fingerprint"], validation_images=len(target_rows),
                    initial_weights_sha256=pretrained_hash, parameters=sizes, settings=settings, seed=args.seed,
                    implementation_sha256=hashlib.sha256((REPO / "src/reem_p3.py").read_bytes()).hexdigest(),
                    runner_sha256=file_sha256(Path(__file__)),
                    variants={})
    path = folder / "comparison.json"
    atomic_json(path, metadata)
    print(f"RUN FOLDER {folder}", flush=True)
    for name in VARIANTS:
        print(f"TRAIN {name}", flush=True)
        run, final_epoch = train(name, folder, settings, args.seed)
        metrics = evaluate(run, target_rows)
        metrics["map50"] = final_epoch["metrics/mAP50(B)"]
        metrics["map50_95"] = final_epoch["metrics/mAP50-95(B)"]
        metrics["weights_sha256"] = file_sha256(run / "weights/last.pt")
        metadata["variants"][name] = metrics
        atomic_json(path, metadata)
        torch.cuda.empty_cache()
        print(f"VALIDATED {name}: {metrics}", flush=True)
    metadata["status"] = "complete"
    atomic_json(path, metadata)


if __name__ == "__main__":
    main()
