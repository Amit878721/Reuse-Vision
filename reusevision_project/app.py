from pathlib import Path
from collections import Counter
from uuid import uuid4
import base64
import io
import os
import time

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = BASE_DIR / "uploads"
MODEL_PATH = WEIGHTS_DIR / "best.pt"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
# These fixed, repeatable thresholds make the experiment comparable between images.
CONFIDENCE_LEVELS = (0.15, 0.25, 0.40, 0.50)
DEFAULT_CONFIDENCE = 0.25

for folder in (WEIGHTS_DIR, RESULTS_DIR, UPLOADS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
model = None
model_error = None


def get_model():
    """Load the local checkpoint once; subsequent requests reuse it."""
    global model, model_error
    if model is not None:
        return model
    if not MODEL_PATH.exists():
        model_error = "Model weights are missing. Run: python download_model.py"
        raise RuntimeError(model_error)
    try:
        model = YOLO(str(MODEL_PATH))
        model_error = None
        return model
    except Exception as exc:
        model_error = f"Could not load the model weights: {exc}"
        raise RuntimeError(model_error) from exc


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def class_names(detector):
    names = detector.names
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def summarise_prediction(prediction, detector):
    """Extract only detections actually emitted by the YOLO model."""
    names = detector.names
    counts = Counter()
    detections = []
    confidences = []
    if prediction.boxes is not None:
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            class_name = str(names[class_id] if isinstance(names, dict) else names[class_id])
            confidence = float(box.conf.item())
            xyxy = [round(float(value), 1) for value in box.xyxy[0].tolist()]
            counts[class_name] += 1
            confidences.append(confidence)
            detections.append({
                "class": class_name,
                "confidence": round(confidence * 100, 1),
                "box": xyxy,
            })
    mean_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0.0
    return {
        "detections": detections,
        "counts": dict(sorted(counts.items())),
        "total": len(detections),
        "mean_confidence": round(mean_confidence, 1),
    }


@app.get("/")
def index():
    try:
        loaded = get_model()
        status = {
            "ready": True,
            "classes": class_names(loaded),
            "message": "Local YOLO checkpoint loaded and ready",
        }
    except Exception as exc:
        status = {"ready": False, "classes": [], "message": str(exc)}
    return render_template(
        "index.html",
        status=status,
        confidence_levels=CONFIDENCE_LEVELS,
        default_confidence=DEFAULT_CONFIDENCE,
    )


@app.post("/detect")
def detect():
    if "image" not in request.files:
        return jsonify(error="Please choose an image first."), 400
    file = request.files["image"]
    if not file or not file.filename:
        return jsonify(error="Please choose an image first."), 400
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type. Use PNG, JPG, JPEG, WEBP, or BMP."), 400

    raw = file.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        return jsonify(error="The selected file is empty."), 400
    if len(raw) > MAX_IMAGE_BYTES:
        return jsonify(error="Image is too large. Maximum size is 12 MB."), 413
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="This file is not a valid image. Please try another one."), 400

    try:
        detector = get_model()
        try:
            requested_confidence = float(request.form.get("confidence", DEFAULT_CONFIDENCE))
        except (TypeError, ValueError):
            return jsonify(error="Choose a valid confidence threshold."), 400

        if not any(abs(requested_confidence - level) < 1e-6 for level in CONFIDENCE_LEVELS):
            return jsonify(error="Choose one of the listed confidence thresholds."), 400

        # Genuine, separate YOLO predictions at each threshold. These values are
        # a sensitivity experiment, not accuracy metrics (no ground-truth labels).
        predictions = {}
        experiment = []
        for threshold in CONFIDENCE_LEVELS:
            started = time.perf_counter()
            prediction = detector.predict(
                source=image,
                conf=threshold,
                imgsz=640,
                verbose=False,
            )[0]
            elapsed_ms = (time.perf_counter() - started) * 1000
            summary = summarise_prediction(prediction, detector)
            predictions[threshold] = prediction
            experiment.append({
                "threshold": threshold,
                "total": summary["total"],
                "mean_confidence": summary["mean_confidence"],
                "inference_ms": round(elapsed_ms, 1),
                "counts": summary["counts"],
            })

        selected_threshold = min(CONFIDENCE_LEVELS, key=lambda level: abs(level - requested_confidence))
        selected_prediction = predictions[selected_threshold]
        selected_summary = summarise_prediction(selected_prediction, detector)

        # Ultralytics plot() returns a BGR ndarray. Convert it for Pillow.
        annotated_bgr = selected_prediction.plot()
        annotated_rgb = Image.fromarray(annotated_bgr[:, :, ::-1])
        output_name = f"{uuid4().hex}.jpg"
        output_path = RESULTS_DIR / output_name
        annotated_rgb.save(output_path, format="JPEG", quality=92, optimize=True)
        with open(output_path, "rb") as result_file:
            annotated_b64 = base64.b64encode(result_file.read()).decode("ascii")

        selected_row = next(row for row in experiment if abs(row["threshold"] - selected_threshold) < 1e-6)
        return jsonify(
            total=selected_summary["total"],
            detections=selected_summary["detections"],
            counts=selected_summary["counts"],
            mean_confidence=selected_summary["mean_confidence"],
            inference_ms=selected_row["inference_ms"],
            selected_confidence=selected_threshold,
            threshold_experiment=experiment,
            experiment_note=(
                "This experiment compares detection counts, mean confidence, and runtime at different confidence thresholds. "
                "It does not calculate accuracy, precision, recall, or mAP because this uploaded image has no ground-truth labels."
            ),
            annotated_image=f"data:image/jpeg;base64,{annotated_b64}",
            download_url=f"/download/{output_name}",
            model_classes=class_names(detector),
        )
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503
    except Exception as exc:
        app.logger.exception("Detection failed")
        return jsonify(error=f"Detection could not be completed: {exc}"), 500


@app.get("/download/<path:filename>")
def download(filename):
    if not filename.endswith(".jpg") or Path(filename).name != filename:
        return jsonify(error="Invalid download request."), 400
    return send_from_directory(
        RESULTS_DIR,
        filename,
        as_attachment=True,
        download_name="reusevision_annotated.jpg",
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Image is too large. Maximum size is 12 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
