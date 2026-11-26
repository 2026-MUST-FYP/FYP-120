"""Screen and verify neutral-start improvements to the original P3 REEM."""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_reem_followup_thresholds import MIN_CONF, measure, predict
from scripts.evaluate_external import file_sha256
from scripts.experiment_identity import atomic_json
from scripts.run_reem_pilot import BASELINE, CONF, DATA, REPO, SETTINGS, SeededDetectionTrainer, load_checkpoint, torch, verify_frozen
from scripts.summarize_reem_followup import FOLLOWUP, PILOT, SEAMS, read
from src.reem_refinements import REFINEMENTS, build_refinement, ridge_profiles

OUTPUT = ROOT / "runs/reem_refinement"
SCREEN_SEEDS = (20261127, 20261128)
CONFIRM_SEEDS = (20261202, 20261203)
ORIGINAL = "original_reem"


def preflight(names, seed):
    source, _ = load_checkpoint(BASELINE)
    original = build_refinement(source, ORIGINAL, seed).eval()
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 3, 128, 128, generator=generator)
    features = torch.randn(2, 64, 17, 19, generator=generator)
    with torch.no_grad():
        baseline_output = original(image)[0]
        line = torch.zeros(1, 1, 17, 17)
        line[..., 8, :] = 1
        step = (line.cumsum(dim=-2) > 0).float()
        assert ridge_profiles(line).max() == 1 and ridge_profiles(step).max() == 0
    for name in names:
        if name == ORIGINAL:
            continue
        detector = build_refinement(source, name, seed).eval()
        with torch.no_grad():
            torch.testing.assert_close(detector(image)[0], baseline_output, rtol=0, atol=0)
        refinement = detector.model[-2].train()
        refinement(features).square().mean().backward()
        new_parameters = set(dict(refinement.named_parameters())) - set(dict(original.model[-2].named_parameters()))
        gradients = [p.grad for key, p in refinement.named_parameters() if key in new_parameters]
        if not gradients or any(grad is None or not torch.isfinite(grad).all() for grad in gradients):
            raise ValueError(f"Invalid new gradients: {name}")
        if not any(grad.count_nonzero() for grad in gradients):
            raise ValueError(f"No new parameter is trainable: {name}")


def train(name, folder, settings, seed):
    class RefinementTrainer(SeededDetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return build_refinement(weights, name, seed, verbose=verbose)

    print(f"TRAIN {seed} {name}", flush=True)
    trainer = RefinementTrainer(overrides=dict(settings, model=str(BASELINE), data=str(DATA),
                                               project=str(folder), name=name, seed=seed))
    trainer.train()
    run = folder / name
    with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checkpoint = run / "weights/last.pt"
    if len(rows) != settings["epochs"] or not checkpoint.is_file():
        raise ValueError(f"Incomplete model: {run}")
    final = {key.strip(): float(value) for key, value in rows[-1].items()}
    return checkpoint, final["metrics/mAP50(B)"], final["metrics/mAP50-95(B)"]


def original_screen(seed, settings, data):
    study = next(item for item in read(FOLLOWUP / "summary.json")["experiments"] if item["seed"] == seed)
    folder = PILOT / study["run"]
    record = read(folder / "comparison.json")
    checkpoint = folder / ORIGINAL / "weights/last.pt"
    metrics = record["variants"][ORIGINAL]
    if (record["status"] != "complete" or record["settings"] != settings or
            record["data_fingerprint"] != data["fingerprint"] or record["initial_weights_sha256"] != file_sha256(BASELINE) or
            record["implementation_sha256"] != file_sha256(REPO / "src/reem_p3.py") or
            metrics["weights_sha256"] != file_sha256(checkpoint)):
        raise ValueError("Original screen reference differs from current training inputs")
    return checkpoint, metrics["map50"], metrics["map50_95"]


def compare(artifacts, targets, seam_tiles):
    original_predictions = predict(artifacts[ORIGINAL][0], targets)
    original = measure(original_predictions, targets, seam_tiles, CONF)
    budget = original["negative_fp"]
    result = {}
    for variant, (checkpoint, map50, map50_95) in artifacts.items():
        predictions = original_predictions if variant == ORIGINAL else predict(checkpoint, targets)
        fixed = measure(predictions, targets, seam_tiles, CONF)
        threshold = CONF
        if variant != ORIGINAL:
            scores = sorted((max(scores, default=0.) for name, (_, scores) in predictions.items()
                             if not targets[name]["boxes"]), reverse=True)
            threshold = max(MIN_CONF, math.nextafter(scores[budget], math.inf)) if budget < len(scores) else MIN_CONF
        matched = measure(predictions, targets, seam_tiles, threshold)
        item = dict(fixed=fixed, matched=matched, map50=map50, map50_95=map50_95,
                    checkpoint=checkpoint.relative_to(ROOT).as_posix(), weights_sha256=file_sha256(checkpoint))
        result[variant] = item
    return result


def rank_candidates(record, candidates):
    ranked = {}
    for variant in candidates:
        small, ap, seams = [], [], []
        for row in record["experiments"].values():
            original, candidate = row[ORIGINAL], row[variant]
            small.append(candidate["matched"]["small_tp"] - original["matched"]["small_tp"])
            seams.append(candidate["matched"]["seam_fp"] - original["matched"]["seam_fp"])
            ap.append(candidate["map50_95"] - original["map50_95"])
        ranked[variant] = dict(small_deltas=small, seam_fp_deltas=seams, map50_95_deltas=ap,
                               eligible=all(x >= 0 for x in small) and sum(small) >= 2 and sum(seams) <= 0 and sum(ap) >= 0)
    return ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "confirm"))
    parser.add_argument("--candidate", choices=tuple(REFINEMENTS))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if (args.stage == "confirm") != bool(args.candidate):
        parser.error("Confirmation requires exactly one candidate, and screening takes none")
    candidates = (args.candidate,) if args.candidate else tuple(REFINEMENTS)
    seeds = SCREEN_SEEDS if args.stage == "screen" else CONFIRM_SEEDS
    epochs = 8 if args.stage == "screen" else 12
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), "This experiment requires the local GPU."
    data = verify_frozen()
    targets = {row["name"]: row for row in data["rows"] if row["split"] == "target_val" and not row["excluded"]}
    seal = read(FOLLOWUP / "protocol.json")["seam_sha256"]
    if seal != file_sha256(SEAMS):
        raise ValueError("Frozen seam sample was changed")
    seam_tiles = {row["name"] for row in read(SEAMS)["rows"] if row["clear_seam"]}
    preflight(candidates, seeds[0])
    print(f"PREFLIGHT OK: {candidates}; validation {len(targets)}, seam sample {len(seam_tiles)}", flush=True)
    if args.check:
        return

    folder = OUTPUT / (args.stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    settings = dict(SETTINGS, epochs=epochs)
    record = dict(status="running", stage=args.stage, candidates=candidates, seeds=seeds, settings=settings,
                  initial_weights_sha256=file_sha256(BASELINE), data_fingerprint=data["fingerprint"],
                  implementation_sha256=file_sha256(REPO / "src/reem_refinements.py"), seam_sha256=seal,
                  runner_sha256=file_sha256(Path(__file__)),
                  screening_rule="Match the original's negative-image FP budget on validation. Require neither seed to lose "
                                 "small-box hits, at least two extra hits across both seeds, no added seam-image alarms in total, "
                                 "and no mean mAP50-95 loss.", experiments={})
    path = folder / "comparison.json"
    atomic_json(path, record)
    print(f"RUN FOLDER {folder}", flush=True)
    for seed in seeds:
        root = folder / str(seed)
        original = original_screen(seed, settings, data) if args.stage == "screen" else train(ORIGINAL, root, settings, seed)
        artifacts = {ORIGINAL: original}
        for name in candidates:
            artifacts[name] = train(name, root, settings, seed)
        record["experiments"][str(seed)] = compare(artifacts, targets, seam_tiles)
        atomic_json(path, record)
        print(f"VALIDATED {seed}: {record['experiments'][str(seed)]}", flush=True)
    record["ranking"] = rank_candidates(record, candidates)
    record["status"] = "complete"
    atomic_json(path, record)
    for name, rank in record["ranking"].items():
        print(f"RESULT {name}: {rank}", flush=True)


if __name__ == "__main__":
    main()
