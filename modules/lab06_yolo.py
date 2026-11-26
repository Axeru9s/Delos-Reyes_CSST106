import os
from functools import lru_cache
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.request import urlopen

import cv2

from .common import LabError, number

MODEL_LOCK = Lock()


@lru_cache(maxsize=1)
def load_model(path):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(path.parent / "ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        from ultralytics import YOLO
    except (ImportError, OSError, RuntimeError) as error:
        raise LabError("YOLO needs Ultralytics and a compatible PyTorch installation. See the About page.") from error
    if not path.is_file():
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
        try:
            with urlopen(url, timeout=30) as remote:
                with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    try:
                        copyfileobj(remote, temporary)
                        temporary.close()
                        temporary_path.replace(path)
                    finally:
                        temporary_path.unlink(missing_ok=True)
        except OSError as error:
            raise LabError("YOLO weights could not be downloaded. See About for manual setup.") from error
    try:
        return YOLO(str(path), task="detect")
    except (OSError, RuntimeError, ValueError) as error:
        raise LabError("YOLO weights could not be loaded. Place the official yolo11n.pt file in models/ and retry.") from error


def run(ctx):
    image = ctx.image(sample="coffee.jpg")
    confidence = number("confidence", 0.25, 0.01, 1, float)
    weights = ctx.root / "models" / "yolo11n.pt"
    with MODEL_LOCK:
        model = load_model(weights)
        try:
            prediction = model.predict(image, conf=confidence, imgsz=640, device="cpu", verbose=False)[0]
        except (OSError, RuntimeError, ValueError) as error:
            raise LabError("YOLO inference failed. Verify the model file and PyTorch installation.") from error
    annotated = image.copy()
    rows = []
    line = max(1, round(max(image.shape[:2]) / 450))
    font = max(0.42, line * 0.42)
    for coords, score, category in zip(prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist(),
                                       prediction.boxes.cls.tolist()):
        x1, y1, x2, y2 = map(round, coords)
        label = prediction.names[int(category)]
        color = (109, 242, 160)
        text = f"{label} {score:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line)
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, line)
        bottom = max(height + 6, y1)
        cv2.rectangle(annotated, (x1, bottom - height - 6), (x1 + width + 6, bottom), color, -1)
        cv2.putText(annotated, text, (x1 + 3, bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (16, 29, 22), line, cv2.LINE_AA)
        rows.append((label, f"{score:.1%}", f"({x1}, {y1}) – ({x2}, {y2})"))
    result = ctx.result
    result.metrics += [("Model", "YOLO11n · COCO"), ("Confidence threshold", f"{confidence:.0%}"),
                       ("Detections", len(rows))]
    result.add_image("original.png", "Original image", image)
    result.add_image("detection.png", "YOLO detections", annotated)
    if rows:
        result.table("Detected objects", ("Object", "Confidence", "Bounding box (px)"), rows)
    else:
        result.notes.append("No objects met the confidence threshold. Try a lower threshold or another image.")
