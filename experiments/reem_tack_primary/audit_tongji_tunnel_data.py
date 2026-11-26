"""Audit the two author-provided Tongji tunnel datasets without changing them."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data/external"
CLASSIFICATION = EXTERNAL / "Image Classification Dataset/dataset-tongji-fy"
DETECTION = EXTERNAL / "Object Detection Dataset/dataset-tongji-fy"
OUT = ROOT / "reports/tongji_tunnel_data"


def image_row(path, subset):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        thumb = np.asarray(rgb.convert("L").resize((32, 32)), dtype=np.float32)
        coefficients = cv2.dct(thumb)[:8, :8].ravel()[1:]
        median = np.median(coefficients)
        bits = int("".join("1" if x > median else "0" for x in coefficients), 2)
    tokens = path.stem.split("_")
    if len(tokens) < 3 or not all(part.isdigit() for part in tokens[:3]):
        raise ValueError(f"Unknown filename convention: {path}")
    return dict(name=path.stem, folder=subset, image=path.relative_to(ROOT).as_posix(), width=rgb.width,
                height=rgb.height, name_group="_".join(tokens[:3]), pixel_sha256=pixels, phash=bits)


def parse_label(path, row):
    tree = ET.parse(path).getroot()
    if Path(tree.findtext("filename", "")).stem != row["name"]:
        raise ValueError(f"Annotation name mismatch: {path}")
    size = tuple(int(tree.findtext(f"size/{axis}")) for axis in ("width", "height"))
    if size != (row["width"], row["height"]):
        raise ValueError(f"Annotation size mismatch: {path}")
    objects, invalid = [], []
    for obj in tree.findall("object"):
        label = obj.findtext("name", "").strip().lower()
        x1, y1, x2, y2 = (int(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax"))
        box = [x1, y1, x2, y2]
        if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            invalid.append(dict(label=label, box=box))
            continue
        objects.append(dict(label=label, box=box))
    return objects, invalid


def review_sheets(rows):
    detection_groups = {r["name_group"] for r in rows if r["folder"] == "detection"}
    background = sorted((r for r in rows if r["folder"] == "background" and r["name_group"] not in detection_groups),
                        key=lambda r: hashlib.sha256(("tongji-seams-v1:" + r["name"]).encode()).hexdigest())
    choices = []
    groups = set()
    for row in background:
        if row["name_group"] in groups:
            continue
        groups.add(row["name_group"])
        choices.append(dict(row, index=len(choices) + 1))
        if len(choices) == 72:
            break
    for start in range(0, len(choices), 24):
        subset = choices[start:start + 24]
        panel = Image.new("RGB", (1200, ((len(subset) + 5)//6)*218), "white")
        draw = ImageDraw.Draw(panel)
        for i, row in enumerate(subset):
            x, y = (i % 6)*200, (i // 6)*218
            with Image.open(ROOT / row["image"]) as image:
                panel.paste(image.convert("RGB").resize((200, 200)), (x, y))
            draw.text((x+4, y+202), f'{row["index"]}  {row["name"]}', fill="black")
        panel.save(OUT / f"background_sheet_{start//24+1}.jpg", quality=93)
    return choices


def main():
    OUT.mkdir(exist_ok=True)
    images = {p.stem: p for p in sorted((DETECTION / "JPEGImages").glob("*.jpg"))}
    xml = {p.stem: p for p in sorted((DETECTION / "Annotations").glob("*.xml"))}
    if not images or images.keys() != xml.keys():
        raise ValueError("Detection images and XML annotations differ")
    rows = [image_row(p, subset) for subset in ("background", "defects")
            for p in sorted((CLASSIFICATION / subset).glob("*.jpg"))]
    issues = []
    for name, path in images.items():
        row = image_row(path, "detection")
        row["objects"], invalid = parse_label(xml[name], row)
        issues.extend(dict(name=name, **issue) for issue in invalid)
        rows.append(row)

    hashes, names = defaultdict(list), defaultdict(list)
    for row in rows:
        key = (row["folder"], row["name"])
        hashes[row["pixel_sha256"]].append(key)
        names[row["name"]].append(key)
    cross_candidates = []
    classes = [r for r in rows if "objects" not in r]
    detections = [r for r in rows if "objects" in r]
    for a in classes:
        for b in detections:
            distance = (a["phash"] ^ b["phash"]).bit_count()
            if distance <= 4:
                cross_candidates.append(dict(classification=[a["folder"], a["name"]], detection=b["name"], distance=distance))
    labels = Counter(obj["label"] for r in detections for obj in r["objects"])
    crack_rows = [r for r in detections if any(obj["label"] == "crack" for obj in r["objects"])]
    small_cracks = sum(min(obj["box"][2]-obj["box"][0], obj["box"][3]-obj["box"][1]) * 512 / max(r["width"], r["height"]) < 64/3
                       for r in crack_rows for obj in r["objects"] if obj["label"] == "crack")
    choices = review_sheets(rows)
    summary = dict(images=dict(Counter(r["folder"] for r in rows)), detection_labels=dict(labels),
                   crack_images=len(crack_rows), crack_name_groups=len({r["name_group"] for r in crack_rows}),
                   small_crack_boxes_512=small_cracks, invalid_boxes=issues,
                   exact_duplicates=[v for v in hashes.values() if len(v) > 1],
                   repeated_names=[v for v in names.values() if len(v) > 1],
                   cross_dataset_phash_candidates=cross_candidates, review_candidates=len(choices))
    for name, payload in (("audit.json", summary), ("inventory.json", rows), ("review_candidates.json", choices)):
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print({k: len(v) if isinstance(v, list) else v for k, v in summary.items()})


if __name__ == "__main__":
    main()
