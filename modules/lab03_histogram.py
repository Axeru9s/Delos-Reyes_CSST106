import cv2
import numpy as np
from matplotlib.figure import Figure
from skimage.exposure import histogram

from .common import choice, gray, number


def run(ctx):
    original = ctx.image(sample="contrast.png")
    image = gray(original)
    operation = choice("operation", "all", {"all", "histogram", "equalize", "clahe", "stretch"})
    result = ctx.result
    result.add_image("original.png", "Original image", original)
    result.add_image("gray.png", "Original grayscale", image)
    variants = [("Original grayscale", image, "#94a9bc")]
    if operation in ("all", "equalize"):
        equalized = cv2.equalizeHist(image)
        result.add_image("equalized.png", "Global equalization", equalized)
        variants.append(("Equalized", equalized, "#49f3ad"))
    if operation in ("all", "clahe"):
        clip = number("clip_limit", 2.0, 0.1, 40, float)
        grid = number("tile_grid", 8, 2, 32)
        enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(image)
        result.add_image("clahe.png", "CLAHE", enhanced, f"Clip limit {clip:g} · {grid} × {grid} tiles")
        variants.append(("CLAHE", enhanced, "#67cafa"))
    if operation in ("all", "stretch"):
        low, high = np.percentile(image, [2, 98])
        if high > low:
            stretched = np.clip((image.astype(float) - low) * 255 / (high - low), 0, 255).astype("uint8")
        else:
            stretched = image.copy()
            result.notes.append("The 2nd and 98th percentiles coincide. Contrast stretching kept the original values.")
        result.add_image("stretch.png", "Contrast stretch", stretched, "Maps the 2nd–98th percentile range to 0–255.")
        variants.append(("Stretched", stretched, "#ffc77d"))
    fig = Figure(figsize=(8.5, 3.5), layout="constrained")
    ax = fig.subplots()
    for name, variant, color in variants:
        counts, intensities = histogram(variant, source_range="dtype")
        ax.plot(intensities, counts, label=name, color=color, lw=1.4)
        result.metrics.append((f"{name} range", f"{variant.min()}–{variant.max()}"))
    ax.set(xlabel="Grayscale intensity", ylabel="Pixel count", xlim=(0, 255), facecolor="#101a25")
    ax.grid(alpha=0.16, color="white")
    ax.legend(facecolor="#182635", labelcolor="white", loc="upper right")
    result.chart("histogram.png", "Intensity histograms", fig)
