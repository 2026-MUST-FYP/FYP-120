"""Add audited Tongji positives to the frozen mixed set and run a paired smoke study."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_tongji_tunnel_data import DETECTION, OUT, image_row, parse_label
from scripts.experiment_identity import atomic_json
from scripts.prepare_v3 import verify_v3
from scripts.run_reem_detail_study import MIXED_DATA, preflight, train
from scripts.run_reem_refinement import BASELINE, FOLLOWUP, ORIGINAL, REPO, SEAMS, SETTINGS, compare, file_sha256, read, torch
from src.reem_detail_fusion import DETAIL_VARIANTS

CANDIDATE = "detail_line_reem"
SEED = 20261213
DATA = ROOT / "data/reem_tongji_v1"
SETTINGS = dict(SETTINGS, epochs=6)


def table(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(mixed):
    cracks = table("裂缝标注清单.csv")
    seams = table("接缝候选清单.csv")
    inventory = {row["image"]: row for row in read(OUT / "inventory.json")}
    if len(cracks) != 98 or len(seams) != 13:
        raise ValueError("Unexpected audited inventory counts")
    seam_names = {r["name_group"] for r in seams}
    seam_pixels = {r["pixel_sha256"] for r in seams}
    for subdir in ("images", "labels"):
        (DATA / subdir).mkdir(parents=True, exist_ok=True)
    new_rows = []
    for item in cracks:
        source = ROOT / item["image"]
        audit = image_row(source, "detection")
        label_file = DETECTION / "Annotations" / f"{source.stem}.xml"
        objects, invalid = parse_label(label_file, audit)
        boxes = [r["box"] for r in objects if r["label"] == "crack"]
        if (invalid or audit["pixel_sha256"] != item["pixel_sha256"] or objects != inventory[item["image"]]["objects"] or
                audit["name_group"] in seam_names or audit["pixel_sha256"] in seam_pixels or
                len(boxes) != int(item["crack_boxes"])):
            raise ValueError(f"Augmentation input no longer matches the audit: {source}")
        image = DATA / "images" / f"tongji_{source.name}"
        label = DATA / "labels" / f"{image.stem}.txt"
        if not image.exists():
            shutil.copy2(source, image)
        if file_sha256(source) != file_sha256(image):
            raise ValueError(f"Copied image differs: {image}")
        width, height = audit["width"], audit["height"]
        content = "".join(f"0 {(x1+x2)/(2*width):.10f} {(y1+y2)/(2*height):.10f} "
                          f"{(x2-x1)/width:.10f} {(y2-y1)/height:.10f}\n" for x1,y1,x2,y2 in boxes)
        write_frozen(label, content)
        new_rows.append(dict(image=image.relative_to(ROOT).as_posix(), source=item["image"],
                             image_sha256=file_sha256(image), xml_sha256=file_sha256(label_file),
                             label_sha256=file_sha256(label), box_count=len(boxes), group=audit["name_group"]))
    if len({r["image_sha256"] for r in new_rows}) != 98 or sum(r["box_count"] for r in new_rows) != 129:
        raise ValueError("Duplicate images or incorrect Tongji box count")
    config = yaml.safe_load(MIXED_DATA.read_text(encoding="utf-8"))
    original = Path(config["train"]).read_text(encoding="utf-8").splitlines()
    listing = DATA / "train.txt"
    write_frozen(listing, "\n".join(original + [(ROOT / row["image"]).as_posix() for row in new_rows]) + "\n")
    config.update(path=DATA.as_posix(), train=listing.as_posix())
    datafile = DATA / "dataset.yaml"
    write_frozen(datafile, yaml.safe_dump(config, sort_keys=False))
    record = dict(base_data_fingerprint=mixed["fingerprint"], train_sha256=file_sha256(listing),
                  yaml_sha256=file_sha256(datafile), original_presentations=len(original),
                  new_images=len(new_rows), train_presentations=len(original)+len(new_rows),
                  held_seam_candidates=len(seams), rows=new_rows)
    manifest = DATA / "manifest.json"
    write_frozen(manifest, json.dumps(record, indent=2, ensure_ascii=False))
    return datafile, record, file_sha256(manifest)


def write_frozen(path, contents):
    if path.exists():
        if path.read_text(encoding="utf-8") != contents:
            raise ValueError(f"Preserve prepared artifact: {path}")
    else:
        path.write_text(contents, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-for", type=Path, help="Train without the Tongji additions for this completed trial.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    parser.add_argument("--variant", choices=tuple(DETAIL_VARIANTS), default=CANDIDATE)
    parser.add_argument("--original-for", type=Path, help="Reuse original REEM from an identical augmented-data run.")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.original_for and args.control_for:
        parser.error("Choose an augmentation or control reference")
    torch.set_num_threads(2)
    settings = dict(SETTINGS, epochs=args.epochs)
    tack, mixed = verify_v3()
    datafile, dataset, digest = prepare(mixed)
    initial_hash = file_sha256(BASELINE)
    model_hash = file_sha256(REPO / "src/reem_detail_fusion.py")
    reference_hash = None
    original = None
    if reference_path := args.control_for or args.original_for:
        reference = read(reference_path)
        expected = dict(status="complete", seed=args.seed, settings=settings, tack_fingerprint=tack["fingerprint"],
                        training_manifest_sha256=digest, initial_weights_sha256=initial_hash,
                        model_source_sha256=model_hash)
        if any(reference.get(key) != value for key, value in expected.items()):
            raise ValueError("The completed augmentation trial does not match this control")
        for item in reference["experiments"].values():
            if file_sha256(ROOT / item["checkpoint"]) != item["weights_sha256"]:
                raise ValueError("An augmentation checkpoint changed")
        reference_hash = file_sha256(reference_path)
    if args.control_for:
        datafile = MIXED_DATA
    if args.original_for:
        item = reference["experiments"][ORIGINAL]
        original = (ROOT / item["checkpoint"], item["map50"], item["map50_95"])
    seal = read(FOLLOWUP / "protocol.json")["seam_sha256"]
    if file_sha256(SEAMS) != seal:
        raise ValueError("TACK seam review changed")
    seam_tiles = {row["name"] for row in read(SEAMS)["rows"] if row["clear_seam"]}
    targets = {row["name"]: row for row in tack["rows"] if row["split"] == "target_val" and not row["excluded"]}
    preflight(args.seed, args.variant)
    prefix = "tongji_control_" if args.control_for else "tongji_"
    run = ROOT / "runs/reem_refinement" / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    path = run / "comparison.json"
    result = dict(status="running", seed=args.seed, settings=settings, tack_fingerprint=tack["fingerprint"],
                  training_manifest_sha256=mixed["fingerprint"] if args.control_for else digest,
                  initial_weights_sha256=initial_hash, model_source_sha256=model_hash,
                  training_presentations=dataset["original_presentations"] if args.control_for else dataset["train_presentations"],
                  new_images=0 if args.control_for else dataset["new_images"],
                  control_for_sha256=reference_hash if args.control_for else None,
                  original_for_sha256=reference_hash if args.original_for else None,
                  validation_images=len(targets))
    atomic_json(path, result)
    print(f"RUN FOLDER {run}", flush=True)
    artifacts = {ORIGINAL: original or train(run, settings, args.seed, ORIGINAL, datafile)}
    artifacts[args.variant] = train(run, settings, args.seed, args.variant, datafile)
    result.update(status="complete", experiments=compare(artifacts, targets, seam_tiles))
    atomic_json(path, result)
    print(f"RESULT {result['experiments']}", flush=True)


if __name__ == "__main__":
    main()
