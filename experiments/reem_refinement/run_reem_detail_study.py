"""Compare P2-detail REEM with the same original REEM and frozen TACK split."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_identity import atomic_json
from scripts.prepare_v3 import verify_v3
from scripts.run_reem_refinement import (
    BASELINE, CONFIRM_SEEDS, DATA, FOLLOWUP, ORIGINAL, OUTPUT, REPO, SCREEN_SEEDS, SEAMS, SETTINGS,
    SeededDetectionTrainer, build, compare, file_sha256, load_checkpoint, original_screen, rank_candidates,
    read, torch, verify_frozen,
)
from src.reem_detail_fusion import DETAIL_VARIANT, DETAIL_VARIANTS, build_detail_detector

MIXED_DATA = ROOT / "data/iteration_v3/balanced.yaml"
MIXED_SEEDS = (20261206, 20261207)


def preflight(seed, variant):
    source, _ = load_checkpoint(BASELINE)
    reference = build(source, ORIGINAL, seed).eval()
    candidate = build_detail_detector(source, seed, variant=variant).eval()
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 3, 128, 128, generator=generator)
    with torch.no_grad():
        torch.testing.assert_close(reference(image)[0], candidate(image)[0], rtol=0, atol=0)
    block = candidate.model[-2].train()
    coarse = torch.randn(2, 64, 17, 19, generator=generator)
    fine = torch.randn(2, 64, 34, 38, generator=generator)
    block((coarse, fine)).square().mean().backward()
    gradient = block.detail_projection.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.count_nonzero():
        raise ValueError("P2 detail projection must receive finite nonzero gradients")


def train(folder, settings, seed, variant=DETAIL_VARIANT, data_path=DATA):
    class DetailTrainer(SeededDetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if variant == ORIGINAL:
                return build(weights, ORIGINAL, seed, verbose)
            return build_detail_detector(weights, seed, verbose, variant)

    print(f"TRAIN {seed} {variant}", flush=True)
    trainer = DetailTrainer(overrides=dict(settings, model=str(BASELINE), data=str(data_path),
                                           project=str(folder), name=variant, seed=seed))
    trainer.train()
    run = folder / variant
    with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checkpoint = run / "weights/last.pt"
    if len(rows) != settings["epochs"] or not checkpoint.is_file():
        raise ValueError(f"Incomplete model: {run}")
    final = {key.strip(): float(value) for key, value in rows[-1].items()}
    return checkpoint, final["metrics/mAP50(B)"], final["metrics/mAP50-95(B)"]


def saved_references(path, settings, fingerprint, seeds, stage, mixed_fingerprint=None):
    archive = read(path)
    if (archive["status"] != "complete" or archive["stage"] != stage or
            archive["settings"] != settings or archive["data_fingerprint"] != fingerprint or
            tuple(archive["seeds"]) != seeds or archive.get("mixed_data_fingerprint") != mixed_fingerprint or
            archive["initial_weights_sha256"] != file_sha256(BASELINE) or
            archive["implementation_sha256"] != file_sha256(REPO / "src/reem_refinements.py")):
        raise ValueError("Longer original REEM references are incompatible")
    references = {}
    for seed, item in archive["experiments"].items():
        original = item[ORIGINAL]
        weights = ROOT / original["checkpoint"]
        if file_sha256(weights) != original["weights_sha256"]:
            raise ValueError(f"Original REEM checkpoint changed: {weights}")
        references[int(seed)] = weights, original["map50"], original["map50_95"]
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "confirm", "mixed"))
    parser.add_argument("--original-confirmation", type=Path)
    parser.add_argument("--mixed-reference", type=Path)
    parser.add_argument("--variant", choices=tuple(DETAIL_VARIANTS), default=DETAIL_VARIANT)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed-pair", nargs=2, type=int)
    args = parser.parse_args()
    if (args.stage == "confirm") != bool(args.original_confirmation):
        parser.error("Confirmation requires a completed original REEM reference record")
    if args.mixed_reference and args.stage != "mixed":
        parser.error("Mixed original REEM references require the mixed stage")
    if args.epochs is not None and (args.epochs < 1 or args.stage != "mixed"):
        parser.error("A positive epoch override is available only for the mixed stage")
    if args.seed_pair and (args.stage != "mixed" or len(set(args.seed_pair)) != 2 or
                           any(not 0 <= seed < 2**32 for seed in args.seed_pair)):
        parser.error("The mixed stage requires two distinct unsigned 32-bit seeds")

    torch.set_num_threads(2)
    mixed = args.stage == "mixed"
    data, mixed_manifest = verify_v3() if mixed else (verify_frozen(), None)
    seeds = tuple(args.seed_pair or {"screen": SCREEN_SEEDS, "confirm": CONFIRM_SEEDS, "mixed": MIXED_SEEDS}[args.stage])
    data_path = MIXED_DATA if mixed else DATA
    settings = dict(SETTINGS, epochs=args.epochs or (12 if args.stage == "confirm" else 8))
    preflight(seeds[0], args.variant)
    seal = read(FOLLOWUP / "protocol.json")["seam_sha256"]
    if seal != file_sha256(SEAMS):
        raise ValueError("Frozen seam sample was changed")
    targets = {row["name"]: row for row in data["rows"] if row["split"] == "target_val" and not row["excluded"]}
    seam_tiles = {row["name"] for row in read(SEAMS)["rows"] if row["clear_seam"]}
    references = {}
    if args.original_confirmation:
        references = saved_references(args.original_confirmation, settings, data["fingerprint"], seeds, "confirm")
    elif args.mixed_reference:
        references = saved_references(args.mixed_reference, settings, data["fingerprint"], seeds, "mixed", mixed_manifest["fingerprint"])
    elif not mixed:
        references = {seed: original_screen(seed, settings, data) for seed in seeds}
    folder = OUTPUT / ("detail_" + args.stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="running", stage=args.stage, candidates=[args.variant], seeds=seeds, settings=settings,
                    initial_weights_sha256=file_sha256(BASELINE), data_fingerprint=data["fingerprint"],
                    implementation_sha256=file_sha256(REPO / "src/reem_refinements.py"), seam_sha256=seal,
                    detail_fusion_sha256=file_sha256(REPO / "src/reem_detail_fusion.py"),
                    training_data=data_path.relative_to(ROOT).as_posix(),
                    mixed_data_fingerprint=mixed_manifest["fingerprint"] if mixed else None,
                    screening_rule="Use each original REEM negative-image FP budget; both seeds must retain small-box hits, "
                                   "gain at least two in total, add no seam-image alarms overall, and retain mean mAP50-95.",
                    runner_sha256=file_sha256(Path(__file__)), experiments={})
    path = folder / "comparison.json"
    atomic_json(path, metadata)
    print(f"RUN FOLDER {folder}", flush=True)
    for seed in seeds:
        root = folder / str(seed)
        original = references[seed] if references else train(root, settings, seed, ORIGINAL, data_path)
        trial = train(root, settings, seed, args.variant, data_path)
        metadata["experiments"][str(seed)] = compare({ORIGINAL: original, args.variant: trial}, targets, seam_tiles)
        atomic_json(path, metadata)
        print(f"VALIDATED {seed}: {metadata['experiments'][str(seed)]}", flush=True)
    metadata["ranking"] = rank_candidates(metadata, (args.variant,))
    metadata["status"] = "complete"
    atomic_json(path, metadata)
    print(f"RESULT {args.variant}: {metadata['ranking'][args.variant]}", flush=True)


if __name__ == "__main__":
    main()
