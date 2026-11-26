from functools import lru_cache
from tarfile import TarError
from threading import Lock

import cv2
import numpy as np
from flask import request
from matplotlib.figure import Figure
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .common import LabError, choice, number

TRAIN_LOCK = Lock()


@lru_cache(maxsize=1)
def load_lfw(directory):
    try:
        dataset = fetch_lfw_people(data_home=directory, min_faces_per_person=70, resize=0.4,
                                   n_retries=1, delay=1.0)
    except (OSError, ValueError, TarError, EOFError) as error:
        raise LabError("LFW could not be loaded. Check your internet connection or existing dataset cache. "
                       "Choose Synthetic practice faces to run the pipeline offline.") from error
    return dataset.images, dataset.target, dataset.target_names


def practice_faces():
    """Generate labeled illustrations for offline pipeline checks."""
    rng = np.random.default_rng(106)
    images, targets = [], []
    for identity in range(6):
        for _ in range(32):
            frame = np.full((50, 40), 34, dtype=np.uint8)
            x = 20 + int(rng.integers(-2, 3))
            y = 24 + int(rng.integers(-2, 3))
            cv2.ellipse(frame, (x, y), (11 + identity % 3, 15 + identity // 3),
                        0, 0, 360, 130 + identity * 11, -1)
            for direction in (-1, 1):
                cv2.circle(frame, (x + direction * (4 + identity % 3), y - 4),
                           2 + identity // 3, 48, -1)
            cv2.line(frame, (x, y - 1), (x + identity % 3 - 1, y + 4), 82, 2)
            cv2.ellipse(frame, (x, y + 6), (4 + identity % 3, 2 + identity // 3),
                        0, 0, 180, 40 + identity * 4, 1)
            noisy = frame.astype(np.float32) + rng.normal(0, 13, frame.shape) + rng.normal(0, 12)
            images.append(np.clip(noisy, 0, 255) / 255)
            targets.append(identity)
    return np.array(images, dtype=np.float32), np.array(targets), [f"Pattern {i + 1}" for i in range(6)]


def run(ctx):
    source = choice("dataset", "lfw", {"lfw", "practice"})
    action = choice("operation", "pca", {"pca", "classify"})
    requested = number("components", 150, 1, 600)
    with TRAIN_LOCK, threadpool_limits(limits=2):
        if source == "lfw":
            images, targets, names = load_lfw(str(ctx.root / "datasets"))
            source_label = "LFW · aligned grayscale · at least 70 images per identity"
        else:
            images, targets, names = practice_faces()
            source_label = "Synthetic practice illustrations"
        data = images.reshape(len(images), -1)
        train, test, y_train, y_test = train_test_split(
            data, targets, test_size=0.25, stratify=targets, random_state=42)
        count = min(requested, len(train) - 1, data.shape[1])
        pca = PCA(n_components=count, whiten=True, svd_solver="randomized", random_state=42)
        train_vectors = pca.fit_transform(train)
        if action == "classify":
            classifier = SVC(kernel="rbf", C=5, class_weight="balanced")
            classifier.fit(train_vectors, y_train)
            predictions = classifier.predict(pca.transform(test))
    result = ctx.result
    if count != requested:
        result.notes.append(f"Used {count} components, the maximum for this training split.")
    if source == "practice":
        result.notes.append("Practice scores measure recognition of generated patterns. Select LFW for the facial recognition report.")
    variance = np.cumsum(pca.explained_variance_ratio_)
    height, width = images.shape[1:]
    result.metrics += [("Dataset", source_label), ("Images", len(images)), ("Identities", len(names)),
                       ("Face size", f"{width} × {height} px"), ("Training / testing", f"{len(train)} / {len(test)}"),
                       ("Components", count), ("Variance explained", f"{variance[-1]:.1%}")]
    fig = Figure(figsize=(8, 2.2), layout="constrained")
    for index, axis in enumerate(fig.subplots(1, min(6, count), squeeze=False)[0]):
        axis.imshow(pca.components_[index].reshape(height, width), cmap="gray")
        axis.set_title(f"PC {index + 1}", fontsize=10)
        axis.axis("off")
    result.chart("eigenfaces.png", "Eigenfaces gallery", fig, "Leading PCA components reshaped into face images.")
    fig = Figure(figsize=(6.5, 3.4), layout="constrained")
    axis = fig.subplots()
    axis.plot(np.arange(1, count + 1), variance * 100, color="#087b69")
    axis.set(xlabel="Number of components", ylabel="Cumulative explained variance (%)", ylim=(0, 101))
    axis.grid(alpha=0.2)
    result.chart("variance.png", "Cumulative explained variance", fig, "PCA was fit on training data only.")
    if action == "classify":
        result.metrics.append(("Test accuracy", f"{accuracy_score(y_test, predictions):.2%}"))
        labels = list(range(len(names)))
        report = classification_report(y_test, predictions, labels=labels,
                                       target_names=list(names), zero_division=0, output_dict=True)
        rows = []
        for label in [*names, "macro avg", "weighted avg"]:
            scores = report[label]
            rows.append((label, *(f"{scores[key]:.3f}" for key in ("precision", "recall", "f1-score")),
                         int(scores["support"])))
        result.table("SVM classification report", ("Identity", "Precision", "Recall", "F1", "Support"), rows)
        if request.form.get("failures") == "on":
            wrong = np.flatnonzero(y_test != predictions)[:6]
            if len(wrong):
                fig = Figure(figsize=(10, 2.6), layout="constrained")
                for index, axis in zip(wrong, fig.subplots(1, len(wrong), squeeze=False)[0]):
                    axis.imshow(test[index].reshape(height, width), cmap="gray")
                    axis.set_title(f"True: {names[y_test[index]]}\nPred: {names[predictions[index]]}", fontsize=7)
                    axis.axis("off")
                result.chart("failures.png", "Misclassified test samples", fig)
            else:
                result.notes.append("The SVM correctly classified every image in this test split.")
