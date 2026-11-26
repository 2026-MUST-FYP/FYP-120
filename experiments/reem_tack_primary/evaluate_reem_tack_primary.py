"""Verify fresh paired runs, then score the predeclared TACK test split."""

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_identity import atomic_json
from scripts.run_reem_refinement import ORIGINAL, file_sha256, measure, predict, read, torch
from scripts.run_reem_pilot import YOLO, SIZE
from scripts.run_reem_tongji_trial import BASELINE, DATA, REPO, SETTINGS, prepare, verify_v3

RUNS = ROOT / "runs/reem_refinement"
STUDY = RUNS / "tack_primary_20261126"


def test_metrics(weights, threshold, targets, seams, project):
    model = YOLO(str(weights))
    measured = model.val(data=str(DATA / "dataset.yaml"), split="test", imgsz=SIZE, batch=16, workers=0, device=0,
                         conf=.001, iou=.5, max_det=300, plots=False, save_json=False, half=False,
                         project=str(project), name=weights.parent.parent.name, verbose=False)
    secondary = measure(predict(weights, targets), targets, seams, threshold)
    return dict(map50=float(measured.box.map50), map50_95=float(measured.box.map), secondary=secondary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--exploratory-test", action="store_true", help="Use the locked exploratory test amendment.")
    args = parser.parse_args()
    study = args.study.resolve()
    if not study.is_relative_to(RUNS.resolve()):
        parser.error("Study directory must be inside the refinement runs")
    torch.set_num_threads(2)
    protocol = read(study / "protocol.json")
    candidate, = [variant for variant in protocol["variants"] if variant != ORIGINAL]
    tack, mixed = verify_v3()
    _, _, manifest_sha = prepare(mixed)
    data = {}
    for path in RUNS.glob("tongji_*/comparison.json"):
        info = read(path)
        if (info.get("new_images") == 98 and info["seed"] in protocol["seeds"] and
                candidate in info.get("experiments", {})):
            if info["seed"] in data:
                raise ValueError(f"Duplicate paired run: {info['seed']}")
            expected = dict(status="complete", validation_images=426, settings=dict(SETTINGS, epochs=protocol["epochs"]), tack_fingerprint=tack["fingerprint"],
                            training_manifest_sha256=manifest_sha, initial_weights_sha256=file_sha256(BASELINE),
                            model_source_sha256=file_sha256(REPO / "src/reem_detail_fusion.py"))
            if any(info.get(key) != value for key, value in expected.items()) or set(info["experiments"]) != set(protocol["variants"]):
                raise ValueError(f"Run does not match the fixed protocol: {path}")
            for trial in info["experiments"].values():
                weights = ROOT / trial["checkpoint"]
                with (weights.parent.parent / "results.csv").open(encoding="utf-8") as f:
                    if len(list(csv.DictReader(f))) != protocol["epochs"]:
                        raise ValueError(f"Wrong epoch count: {weights}")
                if file_sha256(weights) != trial["weights_sha256"]:
                    raise ValueError(f"Checkpoint changed: {weights}")
            data[info["seed"]] = info["experiments"]
    if data.keys() != set(protocol["seeds"]):
        raise ValueError("Missing paired runs")

    gains = {seed: arms[candidate]["map50_95"] - arms[ORIGINAL]["map50_95"] for seed, arms in data.items()}
    result = dict(protocol_sha256=file_sha256(study / "protocol.json"), validation=data, validation_ap_deltas=gains,
                  eligible=all(delta > 0 for delta in gains.values()))
    receipt = study / ("exploratory_test.json" if args.exploratory_test else "result.json")
    if args.exploratory_test:
        amendment = study / "exploratory_protocol.json"
        if read(amendment)["original_validation_receipt_sha256"] != file_sha256(study / "result.json"):
            raise ValueError("The exploratory plan does not reference the original validation receipt")
        if receipt.exists() or sum(gains.values()) <= 0:
            raise ValueError("Preserve existing results or reject a nonpositive average validation gain")
        result.update(exploratory=True, amendment_sha256=file_sha256(amendment))
    atomic_json(receipt, result)
    print("VALIDATION", gains, result["eligible"], flush=True)
    if not result["eligible"] and not args.exploratory_test:
        return

    seam_manifest = read(RUNS / "target_test_seam_review.json")
    targets = {r["name"]: r for r in tack["rows"] if r["split"] == "target_test" and not r["excluded"]}
    seams = {r["name"] for r in seam_manifest["rows"] if r["clear_seam"]}
    if (seam_manifest["split"] != "target_test" or seam_manifest["data_fingerprint"] != tack["fingerprint"] or
            len(targets) != 1898 or len(seams) != 16 or any(targets[name]["boxes"] for name in seams)):
        raise ValueError("Unexpected test inventory")
    result.update(test_images=len(targets), test_seam_images=len(seams), test={})
    for seed, arms in data.items():
        result["test"][seed] = {}
        for variant, trial in arms.items():
            threshold = trial["matched"]["conf"]
            laboratory = test_metrics(ROOT / trial["checkpoint"], threshold, targets, seams, study / "test" / str(seed))
            result["test"][seed][variant] = laboratory
            atomic_json(receipt, result)
            print("TEST", seed, variant, laboratory, flush=True)
    result["test_ap_deltas"] = {seed: arms[candidate]["map50_95"] - arms[ORIGINAL]["map50_95"]
                                for seed, arms in result["test"].items()}
    result["test_complete"] = True
    atomic_json(receipt, result)


if __name__ == "__main__":
    main()
