"""Evaluate ReUseVision's checkpoint on a labelled YOLO-format dataset.

Example:
    python evaluate_model.py --data "C:\\datasets\\motherboard\\data.yaml" --split test

The data YAML must refer to labelled images and use the exact same class names
in the same order as the checkpoint. No evaluation metrics are fabricated.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "weights" / "best.pt"
OUTPUT_ROOT = ROOT / "results" / "evaluation"


def yaml_names(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    names = data.get("names")
    if isinstance(names, dict):
        try:
            return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
        except (TypeError, ValueError):
            return [str(names[key]) for key in sorted(names, key=str)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError("The dataset YAML must define class names under a 'names' key.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local YOLO model on labelled images.")
    parser.add_argument("--data", required=True, help="Path to a YOLO dataset data.yaml")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to best.pt (defaults to project weights)")
    parser.add_argument("--split", choices=("val", "test", "train"), default="test", help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--batch", type=int, default=4, help="Evaluation batch size (default: 4; lower if RAM is low)")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not model_path.is_file():
        print(f"ERROR: model checkpoint not found: {model_path}", file=sys.stderr)
        print("Run 'python download_model.py' first.", file=sys.stderr)
        return 2
    if not data_path.is_file():
        print(f"ERROR: dataset YAML not found: {data_path}", file=sys.stderr)
        return 2
    if args.imgsz < 32 or args.batch < 1:
        print("ERROR: --imgsz must be at least 32 and --batch must be at least 1.", file=sys.stderr)
        return 2

    try:
        model = YOLO(str(model_path))
        model_class_names = [str(name) for name in (model.names[index] for index in sorted(model.names))] if isinstance(model.names, dict) else [str(name) for name in model.names]
        dataset_class_names = yaml_names(data_path)
    except Exception as exc:
        print(f"ERROR: could not read model/dataset configuration: {exc}", file=sys.stderr)
        return 2

    if [name.casefold().strip() for name in model_class_names] != [name.casefold().strip() for name in dataset_class_names]:
        print("ERROR: dataset classes do not match this model's classes in the same order.", file=sys.stderr)
        print(f"Model classes:   {model_class_names}", file=sys.stderr)
        print(f"Dataset classes: {dataset_class_names}", file=sys.stderr)
        print("Do not evaluate until the labels and class order are correct.", file=sys.stderr)
        return 2

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_path}")
    print(f"Split: {args.split}")
    print("Running labelled-dataset evaluation on CPU. This may take a while.")

    try:
        metrics = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device="cpu",
            workers=0,
            plots=True,
            project=str(OUTPUT_ROOT),
            name=f"{args.split}_evaluation",
            exist_ok=True,
            verbose=False,
        )
    except Exception as exc:
        print(f"ERROR: evaluation did not finish: {exc}", file=sys.stderr)
        print("Check that the YAML points to the correct image and label directories.", file=sys.stderr)
        return 1

    # Pull only metrics actually returned by the Ultralytics evaluator.
    raw_metrics = getattr(metrics, "results_dict", {})
    clean_metrics = {}
    for key, value in raw_metrics.items():
        try:
            clean_metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    per_class = []
    try:
        class_map_values = list(metrics.box.maps)
        for class_id, class_name in enumerate(model_class_names):
            if class_id < len(class_map_values):
                per_class.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "map50_95": float(class_map_values[class_id]),
                })
    except Exception:
        per_class = []

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "dataset_yaml": str(data_path),
        "split": args.split,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "class_names": model_class_names,
        "metrics_from_this_run": clean_metrics,
        "per_class_map50_95": per_class,
        "note": "These metrics are from this local evaluation run and depend on the labelled dataset and split supplied by the user.",
    }
    output_dir = OUTPUT_ROOT / f"{args.split}_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "reusevision_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nEVALUATION RESULTS (measured from this run)")
    if not clean_metrics:
        print("Ultralytics returned no summary metric dictionary; inspect its saved evaluation outputs.")
    else:
        for key, value in clean_metrics.items():
            print(f"{key}: {value:.4f}")
    print(f"\nJSON report saved to: {report_path}")
    print(f"Evaluation plots and files saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
