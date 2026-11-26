import cv2

from .common import choice, gray, number


def run(ctx):
    image = ctx.image(sample="calibration.png")
    result = ctx.result
    height, width, channels = image.shape
    result.metrics += [("Width", f"{width} px"), ("Height", f"{height} px"),
                       ("Channels", channels), ("Data type", str(image.dtype))]
    result.add_image("original.png", "Original image", image)
    operation = choice("operation", "info", {"info", "grayscale", "inspect", "pixel",
                                             "crop", "resize", "rotate"})
    if operation == "grayscale":
        result.add_image("grayscale.png", "Grayscale", gray(image), "One intensity channel, 0–255.")
    elif operation in {"inspect", "pixel"}:
        x = number("pixel_x", 0, 0, width - 1)
        y = number("pixel_y", 0, 0, height - 1)
        blue, green, red = map(int, image[y, x])
        result.metrics.append((f"Pixel ({x}, {y})", f"RGB ({red}, {green}, {blue})"))
        if operation == "pixel":
            red, green, blue = (number(name, 0, 0, 255) for name in ("red", "green", "blue"))
            edited = image.copy()
            edited[y, x] = (blue, green, red)
            result.add_image("pixel.png", "Modified pixel", edited,
                             f"Pixel ({x}, {y}) set to RGB ({red}, {green}, {blue}).")
    elif operation == "crop":
        x = number("crop_x", 0, 0, width - 1)
        y = number("crop_y", 0, 0, height - 1)
        w = number("crop_width", min(160, width - x), 1, width - x)
        h = number("crop_height", min(120, height - y), 1, height - y)
        result.add_image("crop.png", "Region of interest", image[y:y+h, x:x+w],
                         f"Origin ({x}, {y}) · {w} × {h} px")
    elif operation == "resize":
        w = number("new_width", 320, 1, 2048)
        h = number("new_height", 240, 1, 2048)
        interpolation = cv2.INTER_AREA if w <= width and h <= height else cv2.INTER_LINEAR
        result.add_image("resize.png", "Resized image",
                         cv2.resize(image, (w, h), interpolation=interpolation), f"{w} × {h} px")
    elif operation == "rotate":
        result.add_image("rotate.png", "Rotated 90° clockwise", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
