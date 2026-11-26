"""Compare P2-detail REEM with the same original REEM and frozen TACK split."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_identity import atomic_json
from scripts.run_reem_refinement import (
    BASELINE, CONFIRM_SEEDS, DATA, FOLLOWUP, ORIGINAL, OUTPUT, REPO, SCREEN_SEEDS, SEAMS, SETTINGS,
    SeededDetectionTrainer, build, compare, file_sha256, load_checkpoint, original_screen, rank_candidates,
    read, torch, verify_frozen,
)
from src.reem_detail_fusion import DETAIL_VARIANT, build_detail_detector


def preflight(seed):
    source, _ = load_checkpoint(BASELINE)
    reference = build(source, ORIGINAL, seed).eval()
    candidate = build_detail_detector(source, seed).eval()
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


def train(folder, settings, seed):
    class DetailTrainer(SeededDetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return build_detail_detector(weights, seed, verbose)

    print(f"TRAIN {seed} {DETAIL_VARIANT}", flush=True)
    trainer = DetailTrainer(overrides=dict(settings, model=str(BASELINE), data=str(DATA),
                                           project=str(folder), name=DETAIL_VARIANT, seed=seed))
    trainer.train()
    run = folder / DETAIL_VARIANT
    with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checkpoint = run / "weights/last.pt"
    if len(rows) != settings["epochs"] or not checkpoint.is_file():
        raise ValueError(f"Incomplete model: {run}")
    final = {key.strip(): float(value) for key, value in rows[-1].items()}
    return checkpoint, final["metrics/mAP50(B)"], final["metrics/mAP50-95(B)"]


def longer_references(path, settings, fingerprint):
    archive = read(path)
    if (archive["status"] != "complete" or archive["stage"] != "confirm" or
            archive["settings"] != settings or archive["data_fingerprint"] != fingerprint or
            tuple(archive["seeds"]) != CONFIRM_SEEDS or
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
    parser.add_argument("stage", choices=("screen", "confirm"))
    parser.add_argument("--original-confirmation", type=Path)
    args = parser.parse_args()
    if (args.stage == "confirm") != bool(args.original_confirmation):
        parser.error("Confirmation requires a completed original REEM reference record")

    torch.set_num_threads(2)
    data = verify_frozen()
    seeds = CONFIRM_SEEDS if args.stage == "confirm" else SCREEN_SEEDS
    settings = dict(SETTINGS, epochs=12 if args.stage == "confirm" else 8)
    preflight(seeds[0])
    seal = read(FOLLOWUP / "protocol.json")["seam_sha256"]
    if seal != file_sha256(SEAMS):
        raise ValueError("Frozen seam sample was changed")
    targets = {row["name"]: row for row in data["rows"] if row["split"] == "target_val" and not row["excluded"]}
    seam_tiles = {row["name"] for row in read(SEAMS)["rows"] if row["clear_seam"]}
    references = (longer_references(args.original_confirmation, settings, data["fingerprint"]) if args.original_confirmation
                  else {seed: original_screen(seed, settings, data) for seed in seeds})
    folder = OUTPUT / ("detail_" + args.stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="running", stage=args.stage, candidates=[DETAIL_VARIANT], seeds=seeds, settings=settings,
                    initial_weights_sha256=file_sha256(BASELINE), data_fingerprint=data["fingerprint"],
                    implementation_sha256=file_sha256(REPO / "src/reem_refinements.py"), seam_sha256=seal,
                    detail_fusion_sha256=file_sha256(REPO / "src/reem_detail_fusion.py"),
                    runner_sha256=file_sha256(Path(__file__)), experiments={})
    path = folder / "comparison.json"
    atomic_json(path, metadata)
    print(f"RUN FOLDER {folder}", flush=True)
    for seed in seeds:
        trial = train(folder / str(seed), settings, seed)
        metadata["experiments"][str(seed)] = compare({ORIGINAL: references[seed], DETAIL_VARIANT: trial}, targets, seam_tiles)
        atomic_json(path, metadata)
        print(f"VALIDATED {seed}: {metadata['experiments'][str(seed)]}", flush=True)
    metadata["ranking"] = rank_candidates(metadata, (DETAIL_VARIANT,))
    metadata["status"] = "complete"
    atomic_json(path, metadata)
    print(f"RESULT {DETAIL_VARIANT}: {metadata['ranking'][DETAIL_VARIANT]}", flush=True)


if __name__ == "__main__":
    main()
