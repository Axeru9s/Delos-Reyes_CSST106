# Computer Vision Laboratory Hub

**CSST106 · Computer Vision and Perception**

A single Flask application for Lab Activity 1 through Lab Activity 7. Each lab
has its own route, parameters, built-in demonstration inputs where applicable,
and downloadable PNG outputs. The dashboard uses an original industrial sci-fi
theme with teal and green accents.

## Quick start

Use Python 3.11 or 3.12. Open a terminal in the extracted project directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m flask --app app run
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` to activate the
environment. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser.

Use the sample checkbox for a quick demonstration, or choose a PNG, JPEG, WebP
or BMP. Each request can contain up to 24 MB of data and each image can contain
up to 12 megapixels. A new upload takes precedence over the sample. Successful
runs save normalized copies of inputs in `uploads/` and generated files in
`outputs/`, grouped by random run IDs. Use the download icon under a result to
save it for the printed report.

On minimal Linux systems, the standard OpenCV wheel may require system libraries
such as `libgl1` and `libglib2.0-0`. If you only need the core stack in a
headless environment, replace `opencv-python` with
`opencv-python-headless`. Keep one OpenCV wheel installed in a given
environment. Ultralytics declares `opencv-python` as a dependency.

## Laboratory activities

| Activity | Route | Demonstration |
| --- | --- | --- |
| 1 · Image Manipulation | `/lab/1` | Image shape and type, RGB pixel inspection and editing, grayscale, ROI crop, resize and clockwise rotation |
| 2 · Edge Detection | `/lab/2` | Gaussian blur, Sobel X and Y, combined magnitude, Canny thresholds |
| 3 · Histogram & Contrast | `/lab/3` | Grayscale histograms, global equalization, CLAHE and percentile stretch |
| 4 · SURF Feature Detection | `/lab/4` | Keypoint visualization and two-image matching with SURF, ORB or SIFT |
| 5 · PCA & Eigenfaces | `/lab/5` | Eigenfaces, cumulative explained variance and an SVM report on held-out faces |
| 6 · YOLO Object Detection | `/lab/6` | YOLO11n image inference with labeled boxes, coordinates and confidence scores |
| 7 · Segmentation Techniques | `/lab/7` | Fixed, Otsu and adaptive thresholds, watershed, contour fill and LR-ASPP segmentation |

The [report checklist](docs/report-checklist.md) uses the same lab names and
numbering as the UI.

## Dataset and model setup

### Lab Activity 4 · SURF

Standard OpenCV wheels generally disable SURF. The page shows the requested
and active detector and runs ORB when SURF cannot be constructed. SIFT can be
selected directly in standard OpenCV builds. For actual SURF results, install
one custom OpenCV Python build with the `opencv_contrib` modules and
`OPENCV_ENABLE_NONFREE=ON`.

### Lab Activity 5 · LFW

Allow internet access for the first LFW run. scikit-learn downloads the aligned
Labeled Faces in the Wild archive (about 233 MiB) and metadata to
`datasets/lfw_home/`, then extracts and caches the dataset. Preload it with

```bash
python -c "from sklearn.datasets import fetch_lfw_people; fetch_lfw_people(data_home='datasets', min_faces_per_person=70, resize=0.4)"
```

If Figshare redirects fail, put the four official files below in
`datasets/lfw_home/` using the indicated filenames, then rerun the command.
scikit-learn extracts the archive from the cache.

| Filename | Source |
| --- | --- |
| `lfw-funneled.tgz` | [Aligned archive](https://ndownloader.figshare.com/files/5976015?download=1) |
| `pairsDevTrain.txt` | [Development train pairs](https://ndownloader.figshare.com/files/5976012?download=1) |
| `pairsDevTest.txt` | [Development test pairs](https://ndownloader.figshare.com/files/5976009?download=1) |
| `pairs.txt` | [Evaluation pairs](https://ndownloader.figshare.com/files/5976006?download=1) |

The app uses a seeded, stratified 75/25 split and fits PCA on training data.
The component setting defaults to 150 and is capped to the training
rank bound. The **Synthetic practice faces (offline)** option runs the same
pipeline on generated patterns. Use LFW runs for facial recognition results in
the report. LFW is a historical benchmark with class and demographic imbalance.
Its scores describe the chosen test split.

### Lab Activities 6 and 7 · Deep-learning dependencies

Install the model dependencies in the same virtual environment.

```bash
python -m pip install -r requirements-models.txt
```

For CPU-only PyTorch wheels, install a compatible torch/torchvision pair using
the [official PyTorch installer](https://pytorch.org/get-started/locally/) before
installing the model requirements. CPU inference was verified with torch 2.8
and torchvision 0.23. Model files are cached under `models/` and load on first
use. For offline environments, download these official weights in advance.

| Lab | File to place in `models/` |
| --- | --- |
| 6 · YOLO11n COCO detector | [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) |
| 7 · LR-ASPP MobileNetV3 Large | [lraspp_mobilenet_v3_large-d234d4ea.pth](https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth) |

The Lab Activity 6 page includes a custom-training example. Lab Activity 7
uses the pretrained model’s supported semantic classes and reports the
percentage of image pixels in each predicted class. Its classical methods
run with the core dependencies.

## Source organization

```text
app.py                    Flask routes, lab metadata and application factory
modules/                  Processing functions and shared media/result helpers
templates/                Jinja layout, dashboard, docs and seven lab pages
static/css/               Responsive theme
static/js/                Input controls, loading feedback and results updates
static/images/            Original dashboard artwork
static/samples/           Bundled demonstration images
static/fonts/             Local web fonts and their license notices
docs/                     Report checklist and asset credits
uploads/ and outputs/     Runtime input and result images
datasets/ and models/     Local datasets and model weights
```

The frontend uses HTML, CSS and JavaScript. Python, Flask, OpenCV, NumPy,
Matplotlib, scikit-image and scikit-learn implement the core workflows.
Ultralytics and PyTorch/torchvision handle pretrained inference.

## Review and GitHub

Start with the bundled samples, run each lab, then try an additional image.
Use the [asset credits](docs/asset-credits.md) when reusing sample images.
The file ignore rules exclude cached data, model weights, generated results
and the virtual environment. Commit the extracted source, including the
`.gitkeep` placeholders, to a new GitHub repository.

The Flask commands above are for local review. Run the app behind a suitable
WSGI server with restricted access if hosting classroom uploads. The app
keeps run files until deleted. Store only images appropriate for your review
environment. Set `CVLAB_DATA_DIR` to put runtime folders on a writable volume.

Webcam capture and in-app custom training are outside the image-based
demonstration. Potential extensions include batch processing, background
training or inference jobs, and configurable result retention.
