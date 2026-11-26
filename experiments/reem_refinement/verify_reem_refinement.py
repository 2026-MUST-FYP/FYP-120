"""Evaluate one confirmed REEM candidate using validation-fixed thresholds."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_reem_refinement import ORIGINAL, OUTPUT, REPO, file_sha256, measure, predict, read, torch, verify_frozen
from scripts.experiment_identity import atomic_json
from scripts.diagnose_v3_sizes import stratum
from scripts.prepare_v3 import verify_v3

MAX_NEGATIVE_FPR_INCREASE = .01


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation", type=Path)
    args = parser.parse_args()
    path = args.confirmation.resolve()
    if not path.is_relative_to(OUTPUT.resolve()) or path.name != "comparison.json":
        parser.error("Expected a confirmation comparison.json inside runs/reem_refinement")
    record = read(path)
    candidate, = record["candidates"]
    if record["stage"] not in ("confirm", "mixed") or record["status"] != "complete" or not record["ranking"][candidate]["eligible"]:
        raise ValueError("Test evaluation requires a completed, eligible validation confirmation")
    output = path.with_name("test_verification.json")
    if output.exists():
        raise ValueError("Preserve the existing test verification")
    if file_sha256(REPO / "src/reem_refinements.py") != record["implementation_sha256"]:
        raise ValueError("The evaluated code differs from the confirmed run")
    if record.get("line_control_sha256") and file_sha256(REPO / "src/reem_line_control.py") != record["line_control_sha256"]:
        raise ValueError("The line-control code differs from the confirmed run")
    if record.get("channel_control_sha256") and file_sha256(REPO / "src/reem_channel_control.py") != record["channel_control_sha256"]:
        raise ValueError("The channel-control code differs from the confirmed run")
    if record.get("detail_fusion_sha256") and file_sha256(REPO / "src/reem_detail_fusion.py") != record["detail_fusion_sha256"]:
        raise ValueError("The detail-fusion code differs from the confirmed run")

    torch.set_num_threads(2)
    if record["stage"] == "mixed":
        data, mixed = verify_v3()
        if mixed["fingerprint"] != record["mixed_data_fingerprint"]:
            raise ValueError("The frozen mixed training set differs from the confirmed run")
    else:
        data = verify_frozen()
    if data["fingerprint"] != record["data_fingerprint"]:
        raise ValueError("The evaluation data differs from the confirmed run")
    splits = {}
    for split in ("target_test", "unseen_tc"):
        targets = {row["name"]: row for row in data["rows"] if row["split"] == split and not row["excluded"]}
        if len(targets) != data["counts"][split]["images"]:
            raise ValueError(f"Incomplete test sample: {split}")
        splits[split] = targets

    result = dict(status="running", selected_candidate=candidate, confirmation_sha256=file_sha256(path),
                  data_fingerprint=data["fingerprint"], max_target_negative_fpr_increase=MAX_NEGATIVE_FPR_INCREASE,
                  splits={}, experiments={})
    for split, targets in splits.items():
        result["splits"][split] = dict(images=len(targets), negatives=sum(not r["boxes"] for r in targets.values()),
                                       small_boxes=sum(stratum(box) == "small" for r in targets.values() for box in r["boxes"]))
    atomic_json(output, result)
    for seed, models in record["experiments"].items():
        results = {}
        for variant, item in models.items():
            checkpoint = ROOT / item["checkpoint"]
            if file_sha256(checkpoint) != item["weights_sha256"]:
                raise ValueError(f"Checkpoint changed: {checkpoint}")
            threshold = item["matched"]["conf"]
            results[variant] = {split: measure(predict(checkpoint, targets), targets, set(), threshold)
                                for split, targets in splits.items()}
            print(seed, variant, results[variant], flush=True)
        result["experiments"][seed] = results
        atomic_json(output, result)

    primary = [models[candidate]["target_test"]["small_tp"] - models[ORIGINAL]["target_test"]["small_tp"]
               for models in result["experiments"].values()]
    fp = [models[candidate]["target_test"]["negative_fp"] - models[ORIGINAL]["target_test"]["negative_fp"]
          for models in result["experiments"].values()]
    mean_fpr = sum(fp) / len(fp) / result["splits"]["target_test"]["negatives"]
    result.update(status="complete", small_box_deltas=primary, mean_negative_fpr_delta=mean_fpr,
                  local_verification_passed=all(delta >= 0 for delta in primary) and sum(primary) >= 2 and
                  mean_fpr <= MAX_NEGATIVE_FPR_INCREASE)
    atomic_json(output, result)
    print("LOCAL VERIFICATION", result["local_verification_passed"], primary, mean_fpr, flush=True)


if __name__ == "__main__":
    main()
