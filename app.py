"""Flask entry point for the CSST106 Computer Vision Laboratory Hub."""
import os
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import matplotlib
from flask import Flask, abort, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

matplotlib.use("Agg")

from modules import (lab01_image_manipulation, lab02_edge_detection, lab03_histogram,
                     lab04_surf_features, lab05_pca, lab06_yolo, lab07_segmentation)
from modules.common import Context, LabError, Result, valid_id

ROOT = Path(__file__).resolve().parent

LABS = {
    1: dict(title="Image Manipulation", short="Image Manipulation", icon="image",
            summary="Inspect pixels, convert color, crop regions, and apply geometric transforms.",
            objective="Explore how digital images are represented as arrays and how basic operations change their shape, orientation, and pixel values.",
            tags=("Pixels", "Transforms"), module=lab01_image_manipulation),
    2: dict(title="Edge Detection", short="Edge Detection", icon="edges",
            summary="Extract gradients and compare Sobel and Canny edge maps.",
            objective="Compare horizontal and vertical intensity derivatives with Canny’s multi-stage edge detection after Gaussian smoothing.",
            tags=("Sobel", "Canny"), module=lab02_edge_detection),
    3: dict(title="Histogram & Contrast", short="Histogram & Contrast", icon="histogram",
            summary="Analyze intensity distributions and recover detail in low-contrast images.",
            objective="Compare grayscale histograms before and after global equalization, local equalization, and percentile-based contrast stretching.",
            tags=("Equalization", "CLAHE"), module=lab03_histogram),
    4: dict(title="SURF Feature Detection", short="SURF Features", icon="features",
            summary="Detect keypoints and match local descriptors across two views.",
            objective="Study repeatable local keypoints and descriptor matching with SURF, ORB, or SIFT.",
            tags=("Keypoints", "Matching"), module=lab04_surf_features),
    5: dict(title="PCA & Eigenfaces", short="PCA & Eigenfaces", icon="face",
            summary="Reduce face-image dimensions and evaluate an SVM classifier.",
            objective="Learn a lower-dimensional space from training faces, visualize its principal components, and measure recognition on a held-out split.",
            tags=("PCA", "SVM"), module=lab05_pca),
    6: dict(title="YOLO Object Detection", short="YOLO Detection", icon="detect",
            summary="Locate and label objects with a pretrained YOLO model.",
            objective="Run image inference and interpret bounding boxes, class labels, and confidence scores from a pretrained YOLO11n detector.",
            tags=("YOLO11", "Bounding boxes"), module=lab06_yolo),
    7: dict(title="Segmentation Techniques", short="Segmentation", icon="segment",
            summary="Compare threshold, watershed, contour, and semantic segmentation.",
            objective="Compare pixel-level partitioning using intensity, region markers, boundaries, and pretrained semantic classes.",
            tags=("Watershed", "Semantic"), module=lab07_segmentation),
}


def create_app(runtime_root=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=64 * 1024 * 1024, MAX_FORM_PARTS=40)
    runtime = Path(runtime_root or os.environ.get("CVLAB_DATA_DIR", ROOT)).resolve()
    for directory in ("uploads", "outputs", "models", "datasets"):
        (runtime / directory).mkdir(parents=True, exist_ok=True)

    @app.context_processor
    def navigation():
        return {"labs": LABS}

    @app.get("/")
    def index():
        return render_template("index.html", page="home")

    @app.get("/about")
    def about():
        return render_template("about.html", page="about")

    @app.route("/lab/<int:number>", methods=["GET", "POST"])
    def lab(number):
        if number not in LABS:
            abort(404)
        meta = LABS[number]
        template = f"labs/lab{number:02}.html"
        if request.method == "GET":
            return render_template(template, number=number, lab=meta)

        try:
            previous_id = request.form.get("previous_id", "")
        except RequestEntityTooLarge:
            return render_template(template, number=number, lab=meta,
                                   error="The request exceeds 64 MB. Use smaller image files."), 413
        run_id = uuid4().hex
        upload_dir = runtime / "uploads" / run_id
        output_dir = runtime / "outputs" / run_id
        upload_dir.mkdir()
        output_dir.mkdir()
        result = Result(output_dir)
        ctx = Context(runtime, ROOT / "static" / "samples", upload_dir, result, previous_id)
        try:
            meta["module"].run(ctx)
        except LabError as error:
            rmtree(upload_dir)
            rmtree(output_dir)
            return render_template(template, number=number, lab=meta, error=str(error),
                                   previous_id=previous_id if valid_id(previous_id) else ""), 422
        except Exception:
            app.logger.exception("Lab %s run failed", number)
            rmtree(upload_dir, ignore_errors=True)
            rmtree(output_dir, ignore_errors=True)
            return render_template(template, number=number, lab=meta,
                                   error="Processing failed. Try another input or check the setup guide.",
                                   previous_id=previous_id if valid_id(previous_id) else ""), 500
        return render_template(template, number=number, lab=meta,
                               result=result, run_id=run_id, previous_id=run_id)

    @app.get("/outputs/<run_id>/<filename>")
    def output(run_id, filename):
        if not valid_id(run_id) or not filename.endswith(".png"):
            abort(404)
        return send_from_directory(runtime / "outputs" / run_id, filename,
                                   download_name=filename, mimetype="image/png")

    @app.errorhandler(404)
    def missing(_):
        return render_template("missing.html"), 404

    @app.after_request
    def browser_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.endpoint == "output":
            response.headers["Cache-Control"] = "private, no-store"
        return response

    return app


app = create_app()
