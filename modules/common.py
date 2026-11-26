"""Media validation and result helpers shared by all seven labs."""
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Lock
from uuid import UUID

import cv2
import numpy as np
from flask import request
from PIL import Image, UnidentifiedImageError

PLOT_LOCK = Lock()
MAX_PIXELS = 50_000_000
PROCESS_MAX_PIXELS = 12_000_000
FORMATS = ("PNG", "JPEG", "WEBP", "BMP")
# Ultralytics may patch Pillow globally to add HEIF support. Keep this validator independent.
open_image = Image.open


class LabError(ValueError):
    """A problem the user can resolve from the lab page or setup guide."""


@dataclass
class Result:
    directory: Path
    images: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    tables: list = field(default_factory=list)

    def add_image(self, filename, title, image, caption=""):
        if not cv2.imwrite(str(self.directory / filename), image):
            raise RuntimeError(f"Could not save {filename}")
        self.images.append((filename, title, caption))

    def chart(self, filename, title, figure, caption=""):
        with PLOT_LOCK:
            figure.savefig(self.directory / filename, dpi=125, bbox_inches="tight",
                           facecolor=figure.get_facecolor())
        self.images.append((filename, title, caption))

    def table(self, title, headers, rows):
        self.tables.append((title, headers, rows))


@dataclass
class Context:
    root: Path
    samples: Path
    upload_dir: Path
    result: Result
    previous_id: str = ""

    def image(self, field="image", sample=None):
        """Prefer a new upload, then a checked sample, then a retained input."""
        upload = request.files.get(field)
        destination = self.upload_dir / f"{field}.png"
        source = ""
        if upload and upload.filename:
            image, resize_note = decode(upload.read())
            source = "Uploaded image"
            if resize_note:
                self.result.notes.append(resize_note)
        elif request.form.get(f"sample_{field}") == "on" and sample:
            image = cv2.imread(str(self.samples / sample))
            source = f"Sample · {sample}"
        elif valid_id(self.previous_id):
            retained = self.upload_dir.parent / self.previous_id / destination.name
            image = cv2.imread(str(retained)) if retained.is_file() else None
            source = "Previous input"
        else:
            image = None
        if image is None:
            raise LabError("Choose an image or enable the sample input."
                           if field == "image" else "Choose a second image or enable the matching sample.")
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError("Could not save the input image.")
        self.result.metrics.append(("Input" if field == "image" else "Second input", source))
        return image


def valid_id(value):
    try:
        return len(value) == 32 and UUID(value).hex == value
    except (ValueError, TypeError):
        return False


def decode(data):
    """Validate and decode an uploaded image, downscaling inputs above PROCESS_MAX_PIXELS."""
    try:
        with open_image(BytesIO(data), formats=FORMATS) as image:
            if image.width * image.height > MAX_PIXELS:
                raise LabError("Keep each image at or below 50 megapixels.")
            image.verify()
        # Decode as BGR and ignore EXIF rotation so analysis uses stored coordinates.
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
        if decoded is None:
            raise LabError("This image could not be decoded. Try saving it again as PNG.")
        h, w = decoded.shape[:2]
        resize_note = ""
        if w * h > PROCESS_MAX_PIXELS:
            scale = (PROCESS_MAX_PIXELS / (w * h)) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            decoded = cv2.resize(decoded, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resize_note = f"Input was automatically downscaled from {w}×{h} to {new_w}×{new_h} for fast processing."
        return decoded, resize_note
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, cv2.error) as error:
        raise LabError("Upload a readable PNG, JPEG, WebP or BMP within the size limit.") from error


def number(name, default, minimum, maximum, type_=int):
    raw = request.form.get(name, "")
    try:
        value = default if raw == "" else type_(raw)
        if minimum <= value <= maximum:
            return value
    except ValueError:
        pass
    label = name.replace("_", " ").capitalize()
    raise LabError(f"{label} must be between {minimum} and {maximum}.")


def choice(name, default, options):
    value = request.form.get(name, default)
    if value not in options:
        raise LabError(f"Choose a valid {name.replace('_', ' ')}.")
    return value


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def odd_size(name, default, minimum, maximum):
    size = number(name, default, minimum, maximum)
    if size % 2 == 0:
        raise LabError(f"{name.replace('_', ' ').capitalize()} must be odd.")
    return size
