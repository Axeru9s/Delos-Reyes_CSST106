import cv2

from .common import choice, gray, number


def detector_for(name, hessian):
    if name == "SURF":
        try:
            return cv2.xfeatures2d.SURF_create(hessianThreshold=hessian), "SURF", cv2.NORM_L2, None
        except (AttributeError, cv2.error):
            return cv2.ORB_create(nfeatures=1000), "ORB", cv2.NORM_HAMMING, (
                "SURF requires an OpenCV build with the contrib module and nonfree features enabled. ORB ran instead.")
    if name == "SIFT":
        try:
            return cv2.SIFT_create(nfeatures=1000), "SIFT", cv2.NORM_L2, None
        except (AttributeError, cv2.error):
            return cv2.ORB_create(nfeatures=1000), "ORB", cv2.NORM_HAMMING, "SIFT is unavailable. ORB ran instead."
    return cv2.ORB_create(nfeatures=1000), "ORB", cv2.NORM_HAMMING, None


def run(ctx):
    image = ctx.image(sample="calibration.png")
    action = choice("operation", "detect", {"detect", "match"})
    requested = choice("detector", "SURF", {"SURF", "ORB", "SIFT"})
    hessian = number("hessian_threshold", 400, 1, 10000)
    detector, active, norm, info = detector_for(requested, hessian)
    points, descriptors = detector.detectAndCompute(gray(image), None)
    result = ctx.result
    result.metrics += [("Requested detector", requested), ("Active detector", active),
                       ("Keypoints in image 1", len(points))]
    if info:
        result.notes.append(info)
    result.add_image("keypoints.png", f"{active} keypoints", cv2.drawKeypoints(
        image, points, None, (155, 245, 50), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
        "Circle size represents scale and the radial line represents orientation.")
    if action == "match":
        second = ctx.image("second", sample="calibration-rotated.png")
        other_points, other_descriptors = detector.detectAndCompute(gray(second), None)
        matches = []
        if descriptors is not None and other_descriptors is not None:
            matches = sorted(cv2.BFMatcher(norm, crossCheck=True).match(descriptors, other_descriptors),
                             key=lambda match: match.distance)
        result.metrics += [("Keypoints in image 2", len(other_points)), ("Mutual matches", len(matches))]
        result.add_image("matches.png", "Feature matches", cv2.drawMatches(
            image, points, second, other_points, matches[:50], None,
            matchColor=(155, 245, 50), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS),
            f"Showing {min(50, len(matches))} of {len(matches)} mutual nearest matches.")
        if not matches:
            result.notes.append("No mutual matches were found. Try images with overlapping textured regions.")
