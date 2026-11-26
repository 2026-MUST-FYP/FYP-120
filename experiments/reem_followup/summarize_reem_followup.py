"""Verify and summarize the two local REEM follow-up seeds."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_v3_sizes import matched_gt_indices, stratum
from scripts.build_v3_cases import chinese_font
from scripts.evaluate_external import file_sha256
from scripts.experiment_identity import atomic_json

PILOT = ROOT / "runs/reem_pilot"
FOLLOWUP = PILOT / "local_followup_20261126"
SEAMS = FOLLOWUP / "seam_review.json"
NAMES = {"baseline": "原版 YOLO11n", "original_reem": "加原版 REEM", "directional_reem": "加改版 REEM"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified_run(seed, protocol, reference):
    candidates = [(path, read(path)) for path in PILOT.glob("*/comparison.json")]
    matches = [(path, record) for path, record in candidates if record["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one run for {seed}; found {len(matches)}")
    path, record = matches[0]
    if record["status"] != "complete" or set(record["variants"]) != set(protocol["variants"]):
        raise ValueError(f"Run incomplete: {path}")
    for key in ("data_fingerprint", "initial_weights_sha256", "parameters", "implementation_sha256"):
        if record[key] != reference[key]:
            raise ValueError(f"Changed first-pilot input: {key}")
    if record["settings"] != dict(reference["settings"], epochs=protocol["epochs"]):
        raise ValueError(f"Different training settings: {path}")
    if record["runner_sha256"] != file_sha256(ROOT / "scripts/run_reem_pilot.py"):
        raise ValueError(f"Different runner identity: {path}")
    return path.parent, record


def compare_validation(folder, record, targets, seam_tiles, size):
    results, matches, examples = {}, {}, {}
    for variant, metrics in record["variants"].items():
        path = folder / variant
        weights = path / "weights/last.pt"
        if file_sha256(weights) != metrics["weights_sha256"]:
            raise ValueError(f"Checkpoint hash differs: {weights}")
        with (path / "results.csv").open(encoding="utf-8", newline="") as handle:
            epoch_count = len(list(csv.DictReader(handle)))
        prediction = read(path / "validation_predictions.json")
        rows = {row["name"]: row for row in prediction["rows"]}
        if len(rows) != len(prediction["rows"]) or rows.keys() != targets.keys() or epoch_count != size:
            raise ValueError(f"Incomplete training or prediction inventory: {path}")
        frames = defaultdict(list)
        for tile in seam_tiles:
            row = rows[tile["name"]]
            if row["gt"] != 0:
                raise ValueError("Seam sample has a ground-truth crack box")
            frames[tile["frame"]].append(row["fp"] > 0)
        small = set()
        for name, row in rows.items():
            boxes = targets[name]["boxes"]
            if not boxes:
                continue
            preds = row["predictions"]
            matched, _ = matched_gt_indices([p["box"] for p in preds], [p["confidence"] for p in preds],
                                            boxes, prediction["conf"])
            small.update((name, i) for i in matched if stratum(boxes[i]) == "small")
        if len(small) != metrics["by_box_size"]["small"]["tp"]:
            raise ValueError("Small-box matches disagree with saved totals")
        matches[variant] = small
        examples[variant] = [tile["name"] for tile in seam_tiles if rows[tile["name"]]["fp"] > 0]
        results[variant] = dict(
            epochs=epoch_count, map50=metrics["map50"], map50_95=metrics["map50_95"],
            small_tp=len(small), small_total=metrics["by_box_size"]["small"]["gt"],
            all_tp=metrics["tp"], all_total=metrics["tp"] + metrics["fn"],
            negative_fp=metrics["negative_fp_images"], negative_total=metrics["negative_images"],
            seam_fp=len(examples[variant]), seam_total=len(seam_tiles),
            seam_frame_fp=sum(any(hits) for hits in frames.values()), seam_frame_total=len(frames),
        )
    original, directional = matches["original_reem"], matches["directional_reem"]
    cases = dict(added=sorted(directional - original), lost=sorted(original - directional), seam_fp=examples)
    return results, cases


def write_report(summary):
    lines = [
        "# REEM 本机复核结果", "",
        "两套新随机种子，每套比较原版 YOLO11n、原版 REEM 和改版 REEM。六次训练均为 8 轮。沿用首轮的 TACK 划分、初始权重、512 输入和训练设置，使用第 8 轮权重在 426 张验证图片上比较。首轮 5 轮结果另存。", "",
        "| 随机种子 | 模型 | 小框检出 | 无裂缝图误报 | 接缝图误报 | 受误报影响的接缝原始帧 | mAP50 | mAP50–95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for study in summary["experiments"]:
        for key, item in study["variants"].items():
            lines.append(f"| {study['seed']} | {NAMES[key]} | {item['small_tp']}/{item['small_total']} | "
                         f"{item['negative_fp']}/{item['negative_total']} | {item['seam_fp']}/{item['seam_total']} | "
                         f"{item['seam_frame_fp']}/{item['seam_frame_total']} | {item['map50']:.2%} | {item['map50_95']:.2%} |")
    deltas = "、".join(f"{delta:+d}" for delta in summary["small_box_hit_deltas"])
    fpr_delta = 100 * summary["mean_negative_fpr_delta"]
    decision = ("通过本机筛选，可安排学校 GPU 做多种子正式对照。" if summary["local_gate_passed"] else
                "本机结果未达到预先设定的筛选条件。先复查增减的检测框和误报图，再决定是否扩大训练。")
    lines.extend(["", f"改版相对原版 REEM 的两组小框检出差分别为 {deltas}，平均无裂缝图误报率变化为 {fpr_delta:+.2f} 个百分点。{decision}", "",
                  f"接缝清单是在打开本轮模型预测前固定的。抽查 {summary['seam_candidate_tiles']} 张无裂缝标注图，筛出来自 {summary['seam_frames']} 个原始帧的 {summary['seam_tiles']} 张明显接缝图。接缝列统计的是这些图片上任何位置出现裂缝误报，帧级指标将同帧切片合并。", "",
                  "小框按 TACK 原图中的标注框短边小于 21.33 像素计算；固定置信度 0.25，匹配 IoU 0.50。此轮为小样本验证集对照，测试集和外部图像留待正式评价。", "",
                  "`protocol.json` 保留筛选规则，`summary.json` 保留两套训练的汇总，`case_index.json` 保留增减的小框及出现误报的图。"])
    (FOLLOWUP / "本机复核报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_examples(experiments, cases, targets):
    from PIL import Image, ImageDraw

    examples = []
    for study in experiments:
        seed = study["seed"]
        index = cases[str(seed)]
        originals = set(index["seam_fp"]["original_reem"])
        variants = set(index["seam_fp"]["directional_reem"])
        seam = sorted(originals ^ variants) or sorted(originals | variants)
        examples.extend((seed, title, lookup[0][0]) for title, lookup in
                        (("改版新增检出", index["added"]), ("改版新增漏检", index["lost"])) if lookup)
        if seam:
            examples.append((seed, "接缝图出现误报", seam[0]))
    if not examples:
        return

    size, left, row_height = 310, 24, 362
    canvas = Image.new("RGB", (left * 4 + size * 3, 106 + len(examples) * row_height), "#f7faf9")
    draw = ImageDraw.Draw(canvas)
    headline, body = chinese_font(26, bold=True), chinese_font(16)
    draw.text((left, 14), "REEM 验证集预测对照", font=headline, fill="#142f33")
    draw.text((left, 56), "绿色为标注框，橙色为模型预测。示例按类别选取，整体表现以汇总表为准。", font=body, fill="#32545a")
    predictions = {}
    for study in experiments:
        for key in NAMES:
            payload = read(PILOT / study["run"] / key / "validation_predictions.json")
            predictions[(study["seed"], key)] = {row["name"]: row["predictions"] for row in payload["rows"]}
    for row_idx, (seed, title, name) in enumerate(examples):
        y = 94 + row_idx * row_height
        draw.text((left, y), f"{title}  ·  {seed}  ·  {name}", font=body, fill="#142f33")
        with Image.open(ROOT / targets[name]["image"]) as src:
            frame = src.convert("RGB").resize((size, size))
        for col_idx, (key, label) in enumerate(NAMES.items()):
            x = left + col_idx * (size + left)
            draw.text((x, y + 25), label, font=body, fill="#142f33")
            canvas.paste(frame, (x, y + 46))
            for color, boxes in (("#24df89", targets[name]["boxes"]),
                                 ("#ff974f", [p["box"] for p in predictions[(seed, key)][name]])):
                for x1, y1, x2, y2 in boxes:
                    scale = size / 512
                    draw.rectangle((x+x1*scale, y+46+y1*scale, x+x2*scale, y+46+y2*scale), outline=color, width=2)
    canvas.save(FOLLOWUP / "验证图例.png")


def main():
    protocol = read(FOLLOWUP / "protocol.json")
    if protocol["seam_sha256"] != file_sha256(SEAMS):
        raise ValueError("Frozen seam review changed")
    review = read(SEAMS)
    seam_tiles = [row for row in review["rows"] if row["clear_seam"]]
    target_rows = read(ROOT / "reports/iteration_v2/data.json")["rows"]
    targets = {row["name"]: row for row in target_rows if row["split"] == "target_val" and not row["excluded"]}
    reference = read(PILOT / "20261126_110614/comparison.json")
    experiments, cases = [], {}
    for seed in protocol["seeds"]:
        folder, record = verified_run(seed, protocol, reference)
        metrics, cases[str(seed)] = compare_validation(folder, record, targets, seam_tiles, protocol["epochs"])
        experiments.append(dict(seed=seed, run=folder.name, variants=metrics))
    small_deltas, fp_deltas = [], []
    for item in experiments:
        old, new = (item["variants"][key] for key in ("original_reem", "directional_reem"))
        small_deltas.append(new["small_tp"] - old["small_tp"])
        fp_deltas.append(new["negative_fp"] / new["negative_total"] - old["negative_fp"] / old["negative_total"])
    summary = dict(protocol=protocol, experiments=experiments, small_box_hit_deltas=small_deltas,
                   mean_negative_fpr_delta=statistics.mean(fp_deltas),
                   local_gate_passed=all(delta > 0 for delta in small_deltas) and statistics.mean(fp_deltas) <= 0,
                   seam_candidate_tiles=review["candidate_count"], seam_tiles=len(seam_tiles),
                   seam_frames=review["clear_seam_frames"], validation_images=len(targets),
                   pilot_status="complete")
    atomic_json(FOLLOWUP / "summary.json", summary)
    atomic_json(FOLLOWUP / "case_index.json", cases)
    write_report(summary)
    write_examples(experiments, cases, targets)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
