from functools import lru_cache
from threading import Lock

import cv2
import numpy as np
from PIL import Image

from .common import LabError, choice, gray, number, odd_size

SEMANTIC_LOCK = Lock()


@lru_cache(maxsize=1)
def load_semantic(directory):
    try:
        import torch
        from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large
    except (ImportError, OSError, RuntimeError) as error:
        raise LabError("Semantic segmentation needs compatible PyTorch and torchvision packages. See About.") from error
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    try:
        model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
        state = torch.hub.load_state_dict_from_url(weights.url, model_dir=str(directory),
                                                  progress=False, check_hash=True, map_location="cpu")
        model.load_state_dict(state)
    except (OSError, RuntimeError, ValueError) as error:
        raise LabError("Semantic weights could not be loaded. See About for the download location.") from error
    return torch, model.eval(), weights.transforms(resize_size=None), weights.meta["categories"]


def semantic(image, ctx):
    with SEMANTIC_LOCK:
        torch, model, transform, categories = load_semantic(ctx.root / "models")
        # Match the pretrained scale without expanding long panoramas unboundedly.
        height, width = image.shape[:2]
        ratio = min(520 / min(height, width), 840 / max(height, width))
        size = (max(24, round(width * ratio)), max(24, round(height * ratio)))
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        scaled = cv2.resize(image, size, interpolation=interpolation)
        pil = Image.fromarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
        try:
            with torch.inference_mode():
                labels = model(transform(pil).unsqueeze(0))["out"][0].argmax(0).cpu().numpy().astype("uint8")
        except RuntimeError as error:
            raise LabError("Semantic inference failed for this image.") from error
    labels = cv2.resize(labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    count = np.bincount(labels.ravel(), minlength=len(categories))
    palette = np.random.default_rng(106).integers(55, 255, (len(categories), 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    colors = palette[labels]
    overlay = image.copy()
    foreground = labels != 0
    overlay[foreground] = cv2.addWeighted(image, 0.45, colors, 0.55, 0)[foreground]
    ctx.result.add_image("semantic.png", "Semantic segmentation", overlay, "Colors identify different object classes.")
    ctx.result.add_image("semantic-mask.png", "Semantic class mask", colors)
    rows = [(categories[i], f"{pixels / labels.size:.1%}",
             "#" + "".join(f"{channel:02x}" for channel in palette[i][::-1]))
            for i, pixels in enumerate(count) if pixels]
    ctx.result.table("Classes in the mask", ("Class", "Pixel area", "Mask color"), rows)
    ctx.result.metrics.append(("Model", "LR-ASPP · MobileNetV3 Large"))


def watershed(image, mono, invert, distance_ratio):
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    smoothed = cv2.GaussianBlur(mono, (5, 5), 0)
    _, binary = cv2.threshold(smoothed, 0, 255, mode | cv2.THRESH_OTSU)
    # Distance peaks seed foreground regions. The dilated mask bounds the unknown rim.
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    background = cv2.dilate(opened, kernel, iterations=2)
    padded = cv2.copyMakeBorder(opened, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    seeds = ((distance > distance.max() * distance_ratio) * 255).astype(np.uint8)
    count, markers = cv2.connectedComponents(seeds)
    markers += 1
    markers[cv2.subtract(background, seeds) > 0] = 0
    cv2.watershed(image, markers)
    mask = np.zeros_like(image)
    palette = np.random.default_rng(106).integers(65, 255, (count + 1, 3), dtype=np.uint8)
    foreground = markers > 1
    mask[foreground] = palette[markers[foreground]]
    overlay = image.copy()
    overlay[foreground] = cv2.addWeighted(image, 0.55, mask, 0.45, 0)[foreground]
    overlay[markers == -1] = (145, 250, 100)
    return overlay, mask, len(np.unique(markers[foreground]))


def run(ctx):
    method = choice("method", "compare", {"compare", "global", "otsu", "adaptive", "watershed", "edge", "semantic"})
    image = ctx.image(sample="astronaut.jpg" if method == "semantic" else "regions.png")
    mono = gray(image)
    if method in {"compare", "adaptive", "watershed", "edge"} and min(mono.shape) < 3:
        raise LabError("This method needs an image at least 3 pixels wide and tall.")
    invert = choice("polarity", "bright", {"bright", "dark"}) == "dark"
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    result = ctx.result
    result.add_image("original.png", "Original image", image)
    if method == "semantic":
        semantic(image, ctx)
        return
    if method in ("compare", "global"):
        threshold = number("threshold", 125, 0, 255)
        _, mask = cv2.threshold(mono, threshold, 255, mode)
        result.add_image("global.png", "Global threshold", mask, f"Fixed cutoff {threshold}.")
    if method in ("compare", "otsu"):
        threshold, mask = cv2.threshold(mono, 0, 255, mode | cv2.THRESH_OTSU)
        result.add_image("otsu.png", "Otsu threshold", mask, f"Computed cutoff {threshold:.0f}.")
    if method in ("compare", "adaptive"):
        window = odd_size("block_size", 31, 3, 101)
        offset = number("offset", 4, -30, 30)
        mask = cv2.adaptiveThreshold(mono, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, window, offset)
        result.add_image("adaptive.png", "Adaptive threshold", mask, f"{window} × {window} local window · offset {offset}")
    if method in ("compare", "watershed"):
        ratio = number("seed_ratio", 0.60, 0.05, 0.95, float)
        overlay, mask, count = watershed(image, mono, invert, ratio)
        result.add_image("watershed.png", "Watershed boundaries", overlay, "Green marks watershed ridges.")
        result.add_image("watershed-mask.png", "Watershed regions", mask)
        result.metrics.append(("Watershed regions", count))
    if method in ("compare", "edge"):
        edges = cv2.Canny(cv2.GaussianBlur(mono, (5, 5), 0), 50, 150)
        joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mono)
        retained = [contour for contour in contours if cv2.contourArea(contour) >= 16]
        cv2.drawContours(mask, retained, -1, 255, cv2.FILLED)
        result.add_image("edges.png", "Canny edges", edges)
        result.add_image("contour-mask.png", "Contour fill", mask, "External contours approximate foreground regions.")
        result.metrics.append(("Retained contours", len(retained)))
