import cv2
import numpy as np

from .common import LabError, gray, number, odd_size


def run(ctx):
    image = gray(ctx.image(sample="calibration.png"))
    lower = number("lower", 50, 0, 255)
    upper = number("upper", 150, 1, 255)
    if lower >= upper:
        raise LabError("The lower Canny threshold must be below the upper threshold.")
    kernel = odd_size("blur_kernel", 5, 1, 31)
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(dx, dy)
    canny = cv2.Canny(blurred, lower, upper, L2gradient=True)
    result = ctx.result
    result.metrics += [("Gaussian kernel", f"{kernel} × {kernel}"), ("Canny thresholds", f"{lower} / {upper}"),
                       ("Canny edge pixels", int(np.count_nonzero(canny)))]
    result.add_image("grayscale.png", "Original grayscale", image)
    result.add_image("sobel.png", "Sobel magnitude", cv2.normalize(magnitude, None, 0, 255,
                     cv2.NORM_MINMAX, dtype=cv2.CV_8U), "Combined Euclidean gradient magnitude.")
    result.add_image("canny.png", "Canny", canny, "Binary edges after hysteresis.")
    result.add_image("blur.png", "Gaussian preprocessing", blurred)
    for name, axis, gradient in (("sobel-x", "X", dx), ("sobel-y", "Y", dy)):
        display = cv2.normalize(np.abs(gradient), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        result.add_image(f"{name}.png", f"Sobel {axis}", display, "Normalized absolute gradient.")
