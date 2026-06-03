"""
digitization.py
===============
Core ECG image-to-signal conversion module.

The public interface is the :class:`ECGImage` class.  Instantiate it with
four pre-loaded YOLO models and the path to an ECG image, then call
:meth:`ECGImage.run_full_pipeline` followed by
:meth:`ECGImage.save_signals_as_csv` (or :meth:`ECGImage.save_signals_as_wfdb`).

Module-level helpers
--------------------
plot_image        — Quick matplotlib preview of a grayscale array.
shadow_removal    — Morphological background subtraction.
line_length       — Euclidean distance between two 2-D points.
parse_layout_from_folder — Parse rows/cols/layout flags from a folder name.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

import numpy as np
import cv2
cv2.setNumThreads(0)   # disable OpenCV's internal thread pool

import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from scipy.interpolate import interp1d
import wfdb
from scipy import signal
from skimage import morphology, segmentation
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr
from skimage.filters import threshold_multiotsu
from concurrent.futures import ThreadPoolExecutor
import re
import pandas as pd
import torch
import time


from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    visualize_results,
)


def plot_image(img, title="Image Plot", size=(12, 12), show_axis=False):
    """Display a grayscale image with matplotlib.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image array (H × W).
    title : str, optional
        Figure title (default ``"Image Plot"``).
    size : tuple[int, int], optional
        Figure size in inches ``(width, height)`` (default ``(12, 12)``).
    show_axis : bool, optional
        Whether to draw axis ticks (default ``False``).
    """
    plt.figure(figsize=size)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    if not show_axis:
        plt.axis('off')
    plt.show()


def shadow_removal(img):
    """Remove uneven illumination / shadow from a grayscale image.

    Uses morphological dilation followed by median blur to estimate the
    background, then subtracts it from the original to yield a
    normalised, shadow-free image.

    Parameters
    ----------
    img : np.ndarray
        Grayscale uint8 image.

    Returns
    -------
    np.ndarray
        Shadow-corrected uint8 image, intensity range [0, 255].
    """
    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 15)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img


def line_length(x1, y1, x2, y2):
    """Return the Euclidean length of a line segment.

    Parameters
    ----------
    x1, y1 : float
        Coordinates of the first endpoint.
    x2, y2 : float
        Coordinates of the second endpoint.

    Returns
    -------
    float
        Length of the segment in the same units as the inputs.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def parse_layout_from_folder(folder_path):
    """Parse ECG layout metadata encoded in a folder name.

    Expected naming convention::

        ecg_signals_<rows>x<cols>_<Rythm|None>_<Cabrera|Normal>

    Parameters
    ----------
    folder_path : str
        Absolute or relative path whose final component encodes the layout.

    Returns
    -------
    layout_key : tuple[int, int, bool] or None
        ``(rows, cols, is_cabrera)``.  ``None`` if the name does not match.
    calibration : bool or None
        ``True`` when a rhythm (calibration) strip is present, ``None`` on
        parse failure.
    """
    base_name = folder_path.split('/')[-1]
    match = re.search(r'ecg_signals_(\d+)x(\d+)_(None|Rythm)_(Cabrera|Normal)', base_name)
    if match:
        rows = int(match.group(1))
        cols = int(match.group(2))
        calibration = (match.group(3) == 'Rythm')
        cabrera_flag = (match.group(4) == 'Cabrera')
        layout_key = (rows, cols, cabrera_flag)
        return layout_key, calibration
    else:
        return None, None


class ECGImage:
    """End-to-end digitizer for a single 12-lead ECG image.

    The class encapsulates every processing stage — from raw pixel
    loading to per-lead voltage time-series export.  All intermediate
    results (masks, bounding boxes, calibration constants, signal grid)
    are stored as instance attributes so that individual stages can be
    inspected or visualised after the pipeline runs.

    Typical usage
    -------------
    >>> ecg = ECGImage(box_model, seg_model, name_model, pulse_model, "ecg.png")
    >>> ecg.run_full_pipeline()
    >>> ecg.save_signals_as_csv("record_001", directory="./output")

    Parameters
    ----------
    box_model : ultralytics.YOLO
        Pre-loaded YOLO model that detects lead bounding boxes.
    segmentation_model : str
        Path to the YOLO segmentation model checkpoint.  Loaded
        internally by ``patched_yolo_infer.MakeCropsDetectThem`` so
        that it can operate on image patches.
    lead_name_model : ultralytics.YOLO
        Pre-loaded YOLO model that classifies lead name labels
        (I, II, III, aVR, aVL, aVF, V1–V6).
    pulse_model : ultralytics.YOLO
        Pre-loaded YOLO model that detects calibration pulse boxes.
    image_path : str
        Path to the ECG image (.png, .jpg, or .jpeg).
    wfdb_path : str, optional
        Path to a WFDB record (without extension) used for ground-truth
        comparison in :meth:`calculate_metrics_ptb`.  Not required for
        normal digitization.

    Attributes set by the pipeline
    --------------------------------
    image : np.ndarray
        Loaded and padded grayscale image.
    processed_image : np.ndarray
        Shadow-removed, blurred version used for all model inference.
    lead_segmentation : list
        ``CombineDetections`` objects from the three segmentation passes.
    mask_image : np.ndarray
        Binary mask (H × W, uint8) combining all segmented lead polygons.
    row_centers : np.ndarray
        Y-pixel positions of detected lead row centres.
    roi : tuple[float, float]
        ``(min_y, max_y)`` bounding the active ECG area.
    lead_bboxes : list[list[float]]
        ``[x1, y1, x2, y2]`` boxes from ``box_model``.
    lead_name_bboxes : list[dict]
        ``{'bbox': [...], 'class_name': str}`` from ``lead_name_model``.
    reference_pulses : list[dict]
        ``{'bbox': [...], 'image': np.ndarray}`` from ``pulse_model``.
    volt_per_pixel : float
        Calibration constant: millivolts per pixel (vertical axis).
    time_per_pixel : float
        Calibration constant: seconds per pixel (horizontal axis).
    is_cabrera : bool
        Whether the ECG uses Cabrera lead ordering.
    has_calibration_pulse : bool
        Whether a rhythm/calibration strip row is present.
    layout : list[list[str]]
        2-D grid of lead names matching the detected row/column layout.
    grid : list[list[dict]]
        Mask grid — each cell contains ``'lead'`` (str) and ``'signal'``
        (np.ndarray of the binary mask slice).
    signal_grid : list[list[dict]]
        After :meth:`extract_signals`: cells additionally contain
        ``'signal'`` as a list of float mV values.
    """

    def __init__(self, box_model, segmentation_model, lead_name_model, pulse_model,
                 image_path, wfdb_path=""):
        self.image_path = image_path          # stored for retry in run_full_pipeline
        self.load_image(image_path)
        self.wfdb_path = wfdb_path

        self.box_model = box_model
        self.segmentation_model = segmentation_model
        self.lead_name_model = lead_name_model
        self.pulse_model = pulse_model

        self.is_cabrera = None
        self.has_calibration_pulse = None

        self.standard_layouts = {
            # Standard layouts
            (12, 1, False): [['I'], ['II'], ['III'], ['aVR'], ['aVL'], ['aVF'],
                             ['V1'], ['V2'], ['V3'], ['V4'], ['V5'], ['V6']],
            (6, 2, False):  [['I', 'V1'], ['II', 'V2'], ['III', 'V3'],
                             ['aVR', 'V4'], ['aVL', 'V5'], ['aVF', 'V6']],
            (4, 3, False):  [['I', 'II', 'III'], ['aVR', 'aVL', 'aVF'],
                             ['V1', 'V2', 'V3'], ['V4', 'V5', 'V6']],
            (3, 4, False):  [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'],
                             ['III', 'aVF', 'V3', 'V6']],
            # Cabrera layouts
            (12, 1, True):  [['aVL'], ['I'], ['aVR'], ['II'], ['aVF'], ['III'],
                             ['V1'], ['V2'], ['V3'], ['V4'], ['V5'], ['V6']],
            (6, 2, True):   [['aVL', 'V1'], ['I', 'V2'], ['aVR', 'V3'],
                             ['II', 'V4'], ['aVF', 'V5'], ['III', 'V6']],
            (4, 3, True):   [['aVL', 'I', 'aVR'], ['II', 'aVF', 'III'],
                             ['V1', 'V2', 'V3'], ['V4', 'V5', 'V6']],
            (3, 4, True):   [['aVL', 'II', 'V1', 'V4'], ['I', 'aVF', 'V2', 'V5'],
                             ['aVR', 'III', 'V3', 'V6']],
        }

    # ------------------------------------------------------------------
    # Image loading & preprocessing
    # ------------------------------------------------------------------

    def load_image(self, path, target_size=2100):
        """Load, resize, and pad an ECG image.

        The image is scaled so that its height equals *target_size* (cubic
        interpolation) and then padded with a 20-pixel white border on all
        sides.  The result is stored in ``self.image``.

        Parameters
        ----------
        path : str
            Path to the image file.
        target_size : int, optional
            Target image height in pixels (default ``2100``).

        Raises
        ------
        FileNotFoundError
            If OpenCV cannot open the file at *path*.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        resample_factor = target_size / img.shape[0]
        img = cv2.resize(img,
                         (int(img.shape[1] * resample_factor),
                          int(img.shape[0] * resample_factor)),
                         interpolation=cv2.INTER_CUBIC)
        self.image = cv2.copyMakeBorder(img, 20, 20, 20, 20,
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def preprocess_image(self):
        """Apply shadow removal and Gaussian smoothing to ``self.image``.

        The result is stored in ``self.processed_image`` and is used as
        input for all subsequent YOLO inference calls.
        """
        rem = shadow_removal(self.image)
        self.processed_image = cv2.GaussianBlur(rem, (3, 3), 0)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_leads(self):
        """Run patched YOLO instance segmentation at three crop scales.

        The segmentation model is applied to overlapping image patches at
        crop factors of 4, 4.5, and 5 (relative to image height).
        Detections from each pass are combined with NMS
        (``nms_threshold=0.5``) and stored in ``self.lead_segmentation``
        as a list of three ``CombineDetections`` objects.
        """
        segmentations = []
        for shape in [4, 4.5, 5]:
            element_crops = MakeCropsDetectThem(
                image=cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR),
                model_path=self.segmentation_model,
                segment=True,
                show_crops=False,
                shape_x=int(self.processed_image.shape[0] // shape),
                shape_y=int(self.processed_image.shape[0] // shape),
                overlap_x=50,
                overlap_y=50,
                conf=0.8,
                iou=0.7,
                classes_list=[0],
            )
            segmentations.append(CombineDetections(element_crops, nms_threshold=0.5))
        self.lead_segmentation = segmentations

    def make_segmentation_mask(self):
        """Rasterise all segmented polygons into a single binary mask.

        Polygons from all three segmentation passes in
        ``self.lead_segmentation`` are filled and merged into a combined
        mask.  A morphological opening (5 × 5 kernel) removes small
        spurious regions.  The result is stored in ``self.mask_image``
        (uint8, values 0 or 255).
        """
        height, width = self.image.shape[:2]
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        for segmentation in self.lead_segmentation:
            for poly in segmentation.filtered_polygons:
                pts = np.array(poly, dtype=np.int32)
                if pts.ndim == 2:
                    pts = [pts]
                cv2.fillPoly(combined_mask, pts, color=255)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.mask_image = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # ------------------------------------------------------------------
    # Row detection
    # ------------------------------------------------------------------

    def find_row_centers(self):
        """Detect the vertical centre of each ECG lead row.

        Projects the binary mask onto the vertical axis (row sums) and
        applies ``scipy.signal.find_peaks`` twice: once to identify all
        peaks, and a second time with an adaptive minimum distance
        (⅔ × mean inter-peak spacing) to consolidate merged rows.

        Sets
        ----
        self.row_centers : np.ndarray
            Y-pixel indices of each row centre.
        self.first_peak_start : int
            Y-pixel where the topmost row's signal begins.
        self.last_peak_end : int
            Y-pixel where the bottommost row's signal ends.
        """
        image_height, image_width = self.mask_image.shape[:2]
        proj = np.sum(self.mask_image, 1)

        height   = (image_width // 10) * 255
        distance = image_height // 30

        peaks, _ = find_peaks(proj, height=height, distance=distance)
        row_centers, _ = find_peaks(
            proj, height=height,
            distance=int(np.mean(np.diff(peaks)) * (2 / 3))
        )
        self.row_centers = row_centers

        if len(peaks) > 0:
            first_peak  = peaks[0]
            start_index = first_peak
            zero_gap    = 0
            for i in range(first_peak - 1, -1, -1):
                if proj[i] == 0:
                    zero_gap += 1
                    if zero_gap > 2:
                        break
                else:
                    zero_gap    = 0
                    start_index = i
            self.first_peak_start = start_index

            last_peak = peaks[-1]
            end_index = last_peak
            zero_gap  = 0
            for i in range(last_peak + 1, image_height):
                if proj[i] == 0:
                    zero_gap += 1
                    if zero_gap > 2:
                        break
                else:
                    zero_gap  = 0
                    end_index = i
            self.last_peak_end = end_index

    def get_roi(self):
        """Compute the vertical region-of-interest (ROI) spanning all lead rows.

        Extends ⅔ of the mean row spacing above the first row centre and
        below the last.  Stores the result as ``self.roi = (min_y, max_y)``.
        """
        spacing = 2 / 3 * np.mean(np.diff(self.row_centers))
        min_y = max(0, self.row_centers[0] - spacing)
        max_y = min(self.image.shape[0], self.row_centers[-1] + spacing)
        self.roi = (min_y, max_y)

    # ------------------------------------------------------------------
    # YOLO inference helpers — GPU-safe wrappers
    # ------------------------------------------------------------------

    def _predict_safe(self, model, image_bgr, **kwargs):
        """Run a YOLO model prediction in a thread-safe manner.

        On CUDA: wraps the call in a dedicated ``torch.cuda.Stream`` and
        synchronises before returning, preventing contention on the
        default stream when multiple threads call this simultaneously.
        On CPU: wraps with ``torch.no_grad()`` only.

        Parameters
        ----------
        model : ultralytics.YOLO
            The detection/classification model to run.
        image_bgr : np.ndarray
            BGR image array as expected by Ultralytics.
        **kwargs
            Forwarded to ``model.predict()`` (e.g. ``conf``, ``iou``).

        Returns
        -------
        list
            Ultralytics ``Results`` objects.
        """
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
            with torch.no_grad():
                with torch.cuda.stream(stream):
                    results = model.predict(image_bgr, verbose=False, **kwargs)
                stream.synchronize()
        else:
            with torch.no_grad():
                results = model.predict(image_bgr, verbose=False, **kwargs)
        return results

    # ------------------------------------------------------------------
    # Timing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _timed(label, fn, *args, **kwargs):
        """Call *fn* with *args*/*kwargs*, print its wall-clock duration, and return its result.

        Parameters
        ----------
        label : str
            Human-readable stage name printed alongside the elapsed time.
        fn : callable
            The stage function to time.
        *args, **kwargs
            Forwarded to *fn*.

        Returns
        -------
        Any
            Whatever *fn* returns.
        """
        t = time.time()
        result = fn(*args, **kwargs)
        print(f"    ⏱ {label}: {time.time() - t:.1f}s", flush=True)
        return result

    # ------------------------------------------------------------------
    # Box & name extraction
    # ------------------------------------------------------------------

    def extract_lead_boxes(self):
        """Detect lead bounding boxes with ``box_model`` and filter to the ROI.

        Runs ``_predict_safe`` on the preprocessed BGR image at confidence
        0.8, then discards boxes whose ``y1``/``y2`` coordinates fall
        outside ``self.roi``.  Sets ``self.lead_bboxes`` as a list of
        ``[x1, y1, x2, y2]`` float lists.
        """
        image_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        results   = self._predict_safe(self.box_model, image_bgr, conf=0.8)

        min_y, max_y = self.roi
        lead_boxes   = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
                if y1 >= min_y and y2 <= max_y:
                    lead_boxes.append([x1, y1, x2, y2])
        self.lead_bboxes = lead_boxes

    def extract_lead_name_boxes(self):
        """Detect and classify lead name labels with ``lead_name_model``.

        Runs at confidence 0.8, filters to the ROI, and stores results in
        ``self.lead_name_bboxes`` as a list of
        ``{'bbox': [x1, y1, x2, y2], 'class_name': str}`` dicts.
        """
        image_bgr = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        results   = self._predict_safe(self.lead_name_model, image_bgr, conf=0.8)

        min_y, max_y = self.roi
        name_boxes   = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
                if y1 >= min_y and y2 <= max_y:
                    cls_id   = int(box.cls.cpu().numpy()[0])
                    cls_name = self.lead_name_model.names[cls_id]
                    name_boxes.append({'bbox': [x1, y1, x2, y2], 'class_name': cls_name})
        self.lead_name_bboxes = name_boxes

    def extract_reference_pulses(self):
        """Detect calibration pulse boxes with ``pulse_model``.

        Runs at confidence 0.7 (lower than lead detection to improve
        recall on small pulses).  For each detection, stores the
        corresponding image crop alongside the bounding-box coordinates.
        Sets ``self.reference_pulses`` as a list of
        ``{'bbox': [x1, y1, x2, y2], 'image': np.ndarray}`` dicts.
        """
        image_bgr  = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        results    = self._predict_safe(self.pulse_model, image_bgr, conf=0.7)

        pulse_boxes = []
        for r in results:
            for box in r.boxes:
                coord = box.xyxy.cpu().numpy()[0].tolist()
                pulse_boxes.append({
                    'image': self.image[int(coord[1]) - 5:int(coord[3]) + 5,
                                        int(coord[0]) - 5:int(coord[2]) + 5],
                    'bbox': coord,
                })
        self.reference_pulses = pulse_boxes

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def visualize_boxes(self, task='Lead name', show_axis=False):
        """Overlay detection bounding boxes on the original image.

        Parameters
        ----------
        task : {'Lead name', 'Lead box', 'Reference pulse'}
            Which set of boxes to draw.
        show_axis : bool, optional
            Whether to display axis ticks (default ``False``).
        """
        img_copy = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)

        if task == 'Lead name':
            if not self.lead_name_bboxes:
                print("Lead name boxes not extracted.")
                return
            for box in self.lead_name_bboxes:
                x1, y1, x2, y2 = map(int, box['bbox'])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_copy, box['class_name'], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        elif task == 'Lead box':
            if not self.lead_bboxes:
                print("Lead boxes not extracted.")
                return
            for bbox in self.lead_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

        elif task == 'Reference pulse':
            if not self.reference_pulses:
                print("Reference pulses not extracted.")
                return
            for bbox in self.reference_pulses:
                x1, y1, x2, y2 = map(int, bbox['bbox'])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

        else:
            print(f"Unknown task: {task}")
            return

        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        if not show_axis:
            plt.axis('off')
        plt.show()

    def visualize_segmentation(self, show_boxes=False, show_axis=False,
                               fill_mask=True, thickness=1):
        """Render all segmented lead polygons on the original image.

        Aggregates results from all three segmentation passes in
        ``self.lead_segmentation`` and passes them to
        ``patched_yolo_infer.visualize_results``.

        Parameters
        ----------
        show_boxes : bool, optional
            Also draw bounding boxes around each polygon (default ``False``).
        show_axis : bool, optional
            Display axis ticks (default ``False``).
        fill_mask : bool, optional
            Fill polygon interiors with a translucent colour (default ``True``).
        thickness : int, optional
            Polygon outline thickness in pixels (default ``1``).
        """
        all_confidences, all_boxes, all_polygons = [], [], []
        all_classes_ids, all_classes_names       = [], []

        for seg in self.lead_segmentation:
            all_confidences.extend(seg.filtered_confidences)
            all_boxes.extend(seg.filtered_boxes)
            all_polygons.extend(seg.filtered_polygons)
            all_classes_ids.extend(seg.filtered_classes_id)
            all_classes_names.extend(seg.filtered_classes_names)

        visualize_results(
            img=cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB),
            confidences=all_confidences,
            boxes=all_boxes,
            polygons=all_polygons,
            classes_ids=all_classes_ids,
            classes_names=all_classes_names,
            segment=True,
            thickness=thickness,
            fill_mask=fill_mask,
            show_boxes=show_boxes,
            show_class=False,
            axis_off=(not show_axis),
        )

    # ------------------------------------------------------------------
    # Calibration (volt/pixel & time/pixel)
    # ------------------------------------------------------------------

    def get_reference_scale(self):
        """Derive pixel-to-physical calibration constants from the calibration pulses.

        For each pulse in ``self.reference_pulses``, the method:

        1. Enhances contrast with CLAHE and binarises with multi-Otsu thresholding.
        2. Detects horizontal and vertical lines via morphological filtering and
           Hough transform to measure the pulse amplitude (voltage) and width (time).
        3. Refines the time calibration using the Line Segment Detector (LSD).

        Sets
        ----
        self.volt_per_pixel : float
            Millivolts per pixel (vertical axis).  Calibrated to the standard
            1 mV / 10 mm calibration pulse.
        self.time_per_pixel : float
            Seconds per pixel (horizontal axis).  Calibrated to 0.2 s between
            the two inner vertical edges of the calibration pulse.

        Raises
        ------
        RuntimeError
            If no valid pulse yields voltage or time measurements.
        """
        voltages, times, dist = [], [], []

        def _line_length(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        for pulse_idx, pulse in enumerate(self.reference_pulses):
            try:
                img = pulse['image']
                if img is None or img.size == 0:
                    continue
                h, w = img.shape
                if h < 10 or w < 10:
                    continue

                clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(img)
                blurred  = cv2.GaussianBlur(enhanced, (3, 3), 0)

                thresholds = threshold_multiotsu(blurred, classes=2)
                regions    = np.digitize(blurred, bins=thresholds)
                binary     = (regions == 0).astype(np.uint8) * 255

                h_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                v_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
                h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
                v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

                h_lines_raw = cv2.HoughLinesP(
                    h_lines_img, rho=1, theta=np.pi / 180,
                    threshold=w // 4, minLineLength=w // 2, maxLineGap=1
                )
                v_lines_raw = cv2.HoughLinesP(
                    v_lines_img, rho=1, theta=np.pi / 180,
                    threshold=h // 4, minLineLength=h // 2, maxLineGap=1
                )

                combined_lines = []
                if h_lines_raw is not None:
                    combined_lines.extend(h_lines_raw)
                if v_lines_raw is not None:
                    combined_lines.extend(v_lines_raw)

                horizontal_lines, vertical_lines = [], []
                for line in combined_lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                    if angle < 10 or angle > 170:
                        horizontal_lines.append((x1, y1, x2, y2))
                    elif 80 < angle < 100:
                        vertical_lines.append((x1, y1, x2, y2))

                if len(vertical_lines) >= 2:
                    longest_v = sorted(vertical_lines,
                                       key=lambda l: _line_length(*l), reverse=True)[:2]
                    xc1 = (longest_v[0][0] + longest_v[0][2]) / 2
                    xc2 = (longest_v[1][0] + longest_v[1][2]) / 2
                    vertical_spacing = abs(xc1 - xc2)

                    volt_vals = sorted([_line_length(*l) for l in vertical_lines],
                                       reverse=True)[:2]
                    if volt_vals:
                        voltages.append(volt_vals)
                    if vertical_spacing > 5:
                        times.append(vertical_spacing)

                # LSD for more precise time calibration
                lsd = cv2.createLineSegmentDetector(refine=2)
                lsd_lines, _, _, _ = lsd.detect(v_lines_img)

                min_length    = h / 3
                angle_tol_rad = np.deg2rad(5)
                filtered      = []

                if lsd_lines is not None:
                    for line in lsd_lines:
                        x1, y1, x2, y2 = line[0]
                        length = np.hypot(x2 - x1, y2 - y1)
                        if length >= min_length:
                            angle = np.arctan2(y2 - y1, x2 - x1)
                            if np.abs(np.abs(angle) - np.pi / 2) <= angle_tol_rad:
                                filtered.append([[x1, y1, x2, y2]])

                if len(filtered) >= 4:
                    filtered.sort(
                        key=lambda l: np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]),
                        reverse=True
                    )
                    top4 = filtered[:4]
                    top4.sort(key=lambda l: (l[0][0] + l[0][2]) / 2)

                    midpoints = []
                    for l1, l2 in [(top4[0], top4[1]), (top4[2], top4[3])]:
                        xc1 = (l1[0][0] + l1[0][2]) / 2
                        xc2 = (l2[0][0] + l2[0][2]) / 2
                        midpoints.append((xc1 + xc2) / 2)

                    dist.append(midpoints[1] - midpoints[0])

            except Exception as e:
                print(f"    ⚠ Skipping pulse {pulse_idx}: {e}", flush=True)
                continue

        if not voltages:
            raise RuntimeError("No valid reference pulses found for voltage calibration.")
        if not dist:
            raise RuntimeError("No valid reference pulses found for time calibration.")

        self.volt_per_pixel = 1 / np.mean(voltages)
        self.time_per_pixel = 0.2 / np.mean(dist)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def make_bounding_box_features(self, box, axis):
        """Extract three positional features from a bounding box along one axis.

        Parameters
        ----------
        box : list[float]
            Bounding box as ``[x1, y1, x2, y2]``.
        axis : {'x', 'y'}
            Axis along which to compute features.

        Returns
        -------
        list[float]
            ``[axis_min, axis_center, axis_max]``.
        """
        if axis == 'y':
            axis_min, axis_max = box[1], box[3]
        else:
            axis_min, axis_max = box[0], box[2]
        axis_center = (axis_min + axis_max) / 2
        return [axis_min, axis_center, axis_max]

    def bounding_boxes_kmeans(self, bounding_boxes, axis='y',
                              k_min=1, k_max=13, return_model=True):
        """Cluster bounding boxes along one axis using K-Means.

        Tries all values of *k* in ``[k_min, k_max]`` and selects the one
        that maximises the silhouette score.  Cluster labels are sorted by
        mean position along *axis* (ascending).

        Parameters
        ----------
        bounding_boxes : list[list[float]]
            List of ``[x1, y1, x2, y2]`` boxes.
        axis : {'x', 'y'}, optional
            Axis along which to cluster (default ``'y'``).
        k_min : int, optional
            Minimum number of clusters (default ``1``).
        k_max : int, optional
            Maximum number of clusters (default ``13``).
        return_model : bool, optional
            If ``True``, also return the fitted ``KMeans`` model and the
            label remapping dict (default ``True``).

        Returns
        -------
        sorted_labels : np.ndarray of int
            Cluster index (sorted) for each input box.
        best_k : int
            Optimal number of clusters.
        sorted_centers : np.ndarray
            Mean positional value of each cluster, sorted ascending.
        label_map : dict
            Mapping from original K-Means label → sorted label.
        best_model : sklearn.cluster.KMeans
            Fitted model (only when ``return_model=True``).

        Raises
        ------
        ValueError
            If *axis* is not ``'x'`` or ``'y'``, or if there are fewer
            boxes than *k_min*.
        """
        if axis not in ('x', 'y'):
            raise ValueError("Axis must be 'x' or 'y'")
        if len(bounding_boxes) < k_min:
            raise ValueError("Not enough bounding boxes to cluster")

        features = np.array([self.make_bounding_box_features(b, axis)
                             for b in bounding_boxes])

        best_score, best_k = -1, k_min
        best_labels = best_centers = best_model = None

        for k in range(k_min, min(k_max + 1, len(bounding_boxes))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(features)
            score  = silhouette_score(features, labels)
            if score > best_score:
                best_score   = score
                best_k       = k
                best_labels  = labels
                best_centers = kmeans.cluster_centers_
                best_model   = kmeans

        cluster_avgs   = best_centers.mean(axis=1)
        sorted_indices = np.argsort(cluster_avgs)
        label_map      = {old: new for new, old in enumerate(sorted_indices)}
        sorted_labels  = np.array([label_map[l] for l in best_labels])
        sorted_centers = cluster_avgs[sorted_indices]

        if return_model:
            return sorted_labels, best_k, sorted_centers, label_map, best_model
        return sorted_labels, best_k, sorted_centers, label_map

    def check_cabrera(self, num_rows, num_cols):
        """Determine whether the ECG uses Cabrera lead ordering.

        The heuristic varies by layout:

        * **12- or 6-row layouts:** compares the vertical spacing of
          augmented-limb leads (aVR/aVL/aVF) against precordial leads.
          Cabrera re-orders the limb leads so their spacing differs.
        * **4-row, 3-col / 5-row layouts:** checks the vertical spread
          of augmented leads (high std → interleaved → Cabrera).
        * **3-row / 4-col layouts:** checks the horizontal spread of
          augmented leads.

        Sets ``self.is_cabrera`` as a side-effect.

        Parameters
        ----------
        num_rows : int
            Number of detected lead rows.
        num_cols : int
            Number of detected lead columns.

        Returns
        -------
        bool
            ``True`` if Cabrera ordering is detected.
        """
        if num_rows in [13, 12, 7, 6]:
            av_leads = [b for b in self.lead_name_bboxes
                        if b['class_name'] in {'aVR', 'aVL', 'aVF'}]
            v_leads  = [b for b in self.lead_name_bboxes
                        if b['class_name'] in {'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}]

            if not av_leads or not v_leads:
                return False

            y_v  = sorted([(b['bbox'][1] + b['bbox'][3]) / 2 for b in v_leads])
            y_av = sorted([(b['bbox'][1] + b['bbox'][3]) / 2 for b in av_leads])

            threshold = 30
            diff_v    = np.diff(y_v)
            diff_av   = np.diff(y_av)
            diff_v    = diff_v[np.abs(diff_v) > threshold]
            diff_av   = diff_av[np.abs(diff_av) > threshold]

            if len(diff_v) == 0 or len(diff_av) == 0:
                return False

            if abs(np.min(diff_v) - np.min(diff_av)) > (0.25 * np.min(diff_v)):
                self.is_cabrera = True
                return True
            self.is_cabrera = False
            return False

        elif (num_rows == 4 and num_cols == 3) or num_rows == 5:
            av_leads = [b for b in self.lead_name_bboxes
                        if b['class_name'] in {'aVR', 'aVL', 'aVF'}]
            if not av_leads:
                return False
            y_coords = [(b['bbox'][1] + b['bbox'][3]) / 2 for b in av_leads]
            self.is_cabrera = np.std(y_coords) > 25
            return self.is_cabrera

        elif (num_rows == 4 and num_cols == 4) or num_rows == 3:
            av_leads = [b for b in self.lead_name_bboxes
                        if b['class_name'] in {'aVR', 'aVL', 'aVF'}]
            if not av_leads:
                return False
            x_coords = [(b['bbox'][0] + b['bbox'][2]) / 2 for b in av_leads]
            self.is_cabrera = np.std(x_coords) > 25
            return self.is_cabrera

    def get_layout(self, num_rows):
        """Map the detected row count to a standard ECG layout.

        Calls :meth:`check_cabrera` to determine lead ordering, sets
        ``self.layout`` (2-D list of lead name strings) and
        ``self.has_calibration_pulse``, and returns the number of columns.

        Supported row counts
        --------------------
        * 13 → 12×1 with calibration pulse
        * 12 → 12×1 without calibration pulse
        * 7  → 6×2 with calibration pulse
        * 6  → 6×2 without calibration pulse
        * 5  → 4×3 with calibration pulse
        * 4  → 3×4 or 4×3 (disambiguated by V-lead spatial distribution)
        * 3  → 3×4 without calibration pulse

        Parameters
        ----------
        num_rows : int
            Number of detected lead rows.

        Returns
        -------
        int
            Number of lead columns in the detected layout.
        """
        if num_rows == 13:
            num_cols = 1
            self.has_calibration_pulse = True
            cabrera = self.check_cabrera(num_rows, num_cols)
            self.layout = self.standard_layouts[(12, 1, cabrera)]

        elif num_rows == 12:
            num_cols = 1
            self.has_calibration_pulse = False
            cabrera = self.check_cabrera(num_rows, num_cols)
            self.layout = self.standard_layouts[(12, 1, cabrera)]

        elif num_rows == 7:
            num_cols = 2
            self.has_calibration_pulse = True
            cabrera = self.check_cabrera(num_rows, num_cols)
            self.layout = self.standard_layouts[(6, 2, cabrera)]

        elif num_rows == 6:
            num_cols = 2
            self.has_calibration_pulse = False
            cabrera = self.check_cabrera(num_rows, num_cols)
            self.layout = self.standard_layouts[(6, 2, cabrera)]

        elif num_rows == 5:
            num_cols = 3
            self.has_calibration_pulse = True
            cabrera = self.check_cabrera(num_rows, num_cols)
            self.layout = self.standard_layouts[(4, 3, cabrera)]

        elif num_rows == 4:
            v_leads1 = [b for b in self.lead_name_bboxes
                        if b['class_name'] in {'V1', 'V2', 'V3'}]
            v_leads2 = [b for b in self.lead_name_bboxes
                        if b['class_name'] in {'V4', 'V5', 'V6'}]

            def centers(boxes, axis):
                if axis == 'x':
                    return [(b['bbox'][0] + b['bbox'][2]) / 2 for b in boxes]
                return [(b['bbox'][1] + b['bbox'][3]) / 2 for b in boxes]

            x_std1 = np.std(centers(v_leads1, 'x'))
            y_std1 = np.std(centers(v_leads1, 'y'))
            x_std2 = np.std(centers(v_leads2, 'x'))
            y_std2 = np.std(centers(v_leads2, 'y'))

            if x_std1 < y_std1 and x_std2 < y_std2:
                num_cols = 4
                self.has_calibration_pulse = True
            elif y_std1 < x_std1 and y_std2 < x_std2:
                num_cols = 3
                self.has_calibration_pulse = False
            else:
                num_cols = 4
                self.has_calibration_pulse = True

            cabrera = self.check_cabrera(num_rows, num_cols)
            key = (3, 4, cabrera) if num_cols == 4 else (4, 3, cabrera)
            self.layout = self.standard_layouts[key]

        elif num_rows == 3:
            num_cols = 4
            self.has_calibration_pulse = False
            cabrera = self.check_cabrera(num_rows, num_cols)
            self.layout = self.standard_layouts[(3, 4, cabrera)]

        return num_cols

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def make_grid(self, padding=0):
        """Build the lead mask grid from segmentation polygons and lead boxes.

        Steps:

        1. Assigns each segmented polygon to the nearest row centre by
           polygon centroid Y-coordinate.
        2. Crops each row mask to its vertical extent.
        3. Clusters lead bounding boxes into *num_cols* columns with K-Means.
        4. Slices each (row, col) cell from the corresponding row mask.
        5. Applies a morphological opening to each cell slice.

        Sets
        ----
        self.grid : list[list[dict]]
            ``grid[row][col]`` contains ``{'lead': str, 'signal': np.ndarray}``,
            where ``'signal'`` is currently a binary mask slice.
        self.baseline : list[float]
            Relative Y-position of the isoelectric baseline within each row
            (pixels from the top of the cropped row mask).
        self.row : dict[int, np.ndarray]
            Cropped binary mask for each row index.

        Parameters
        ----------
        padding : int, optional
            Extra pixel padding (currently unused, reserved for future use).
        """
        image_height, image_width = self.image.shape[:2]
        num_rows = len(self.row_centers)
        num_cols = self.get_layout(num_rows)

        row_masks    = {i: np.zeros((image_height, image_width), dtype=np.uint8)
                        for i in range(num_rows)}
        row_polygons = {i: [] for i in range(num_rows)}
        row_limits   = {i: [] for i in range(num_rows)}

        for seg in self.lead_segmentation:
            for box, polygon in zip(seg.filtered_boxes, seg.filtered_polygons):
                poly_np   = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                temp_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [poly_np], color=1)

                temp_proj      = np.sum(temp_mask, axis=1)
                total_weight_y = np.sum(temp_proj)
                if total_weight_y == 0:
                    continue

                centroid_y = int(np.sum(np.arange(temp_proj.shape[0]) * temp_proj)
                                 / total_weight_y)

                y_vals      = [pt[1] for pt in polygon]
                min_y, max_y = min(y_vals), max(y_vals)

                if max_y < self.first_peak_start or min_y > self.last_peak_end:
                    continue

                closest_idx = int(np.argmin(np.abs(self.row_centers - centroid_y)))

                if self.has_calibration_pulse and closest_idx == num_rows - 1:
                    continue

                row_polygons[closest_idx].append(polygon)
                cv2.fillPoly(row_masks[closest_idx], [poly_np], color=1)

                if not row_limits[closest_idx]:
                    row_limits[closest_idx] = [min_y, max_y]
                else:
                    row_limits[closest_idx][0] = min(row_limits[closest_idx][0], min_y)
                    row_limits[closest_idx][1] = max(row_limits[closest_idx][1], max_y)

        cropped_row_masks = {}
        for i in range(num_rows):
            if self.has_calibration_pulse and i == num_rows - 1:
                continue
            if not row_limits[i]:
                continue
            min_y, max_y = row_limits[i]
            cropped_row_masks[i] = row_masks[i][min_y:max_y + 1, :]

        lead_boxes = [
            box for box in self.lead_bboxes
            if not (
                self.has_calibration_pulse and
                int(np.argmin(np.abs(self.row_centers - (box[1] + box[3]) / 2)))
                == num_rows - 1
            )
        ] if self.has_calibration_pulse else self.lead_bboxes

        self.row = cropped_row_masks

        if num_cols != 1:
            labels_cols, _, _, _ = self.bounding_boxes_kmeans(
                lead_boxes, axis='x', k_min=num_cols, k_max=num_cols, return_model=False
            )
        else:
            labels_cols = np.zeros(len(lead_boxes), dtype=int)

        boxes_arr = np.array(lead_boxes)
        min_x_per_col, max_x_per_col = [], []
        for col_label in range(num_cols):
            col_boxes = boxes_arr[labels_cols == col_label]
            min_x_per_col.append(col_boxes[:, 0].min())
            max_x_per_col.append(col_boxes[:, 2].max())

        mask_grid, relative_baselines = [], []
        for row_idx, row_slice in cropped_row_masks.items():
            row_cells = []
            for col_idx in range(num_cols):
                x_min      = max(0, int(min_x_per_col[col_idx]))
                x_max      = min(image_width, int(max_x_per_col[col_idx]))
                cell_slice = row_slice[:, x_min:x_max]

                kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                cell_slice = cv2.morphologyEx(cell_slice, cv2.MORPH_OPEN, kernel)

                row_cells.append({
                    'lead':   self.layout[row_idx][col_idx],
                    'signal': cell_slice,
                })

            relative_baselines.append(
                self.row_centers[row_idx] - row_limits[row_idx][0]
            )
            mask_grid.append(row_cells)

        self.grid     = mask_grid
        self.baseline = relative_baselines

    def visualize_grid(self, figsize=(15, 10)):
        """Display the lead mask grid — one subplot per (row, col) cell.

        Parameters
        ----------
        figsize : tuple[int, int], optional
            Overall figure size in inches (default ``(15, 10)``).

        Raises
        ------
        ValueError
            If :meth:`make_grid` has not been called yet.
        """
        if not hasattr(self, 'grid') or not self.grid:
            raise ValueError("Grid not generated. Call make_grid() first.")

        num_rows = len(self.grid)
        num_cols = len(self.grid[0]) if isinstance(self.grid[0], list) else 1

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_cols == 1:
            axes = np.atleast_2d(axes).T
        elif num_rows == 1:
            axes = np.atleast_2d(axes)

        for row_idx in range(num_rows):
            row = self.grid[row_idx] if isinstance(self.grid[row_idx], list) \
                  else [self.grid[row_idx]]
            for col_idx, cell in enumerate(row):
                ax = axes[row_idx][col_idx]
                ax.imshow(cell['signal'], cmap='gray', aspect='auto')
                ax.set_title(cell['lead'], fontsize=10)
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def binarize_signal(self, img, window_length=11, polyorder=2):
        """Extract a 1-D signal trace from a binary lead mask column by column.

        For each image column, computes the intensity-weighted centroid of
        non-zero pixels.  Uses the second derivative of the initial centroid
        trace (Savitzky-Golay filtered) to disambiguate ambiguous columns:
        upward concavity → take the topmost pixels; downward concavity →
        take the bottommost pixels; otherwise use all non-zero pixels.

        Parameters
        ----------
        img : np.ndarray
            Binary uint8 mask for a single lead cell (H × W).
        window_length : int, optional
            Savitzky-Golay filter window (must be odd, default ``11``).
        polyorder : int, optional
            Savitzky-Golay polynomial order (default ``2``).

        Returns
        -------
        x_coords : np.ndarray of int
            Column indices where a signal was detected.
        final_signal : np.ndarray of float
            Weighted-centroid Y-coordinate for each column in *x_coords*.
        """
        height, width = img.shape
        x_coords = []
        initial_signal = []

        for col in range(width):
            column = img[:, col]
            if np.sum(column) == 0:
                continue

            y_indices = np.arange(height)
            weights = column.astype(float)
            centroid = np.average(y_indices, weights=weights)
            initial_signal.append(centroid)
            x_coords.append(col)

        x_coords = np.array(x_coords)
        initial_signal = np.array(initial_signal)

        second_deriv = np.gradient(np.gradient(savgol_filter(initial_signal, window_length, polyorder)))
        second_deriv = savgol_filter(second_deriv, 11, 3)

        x_out, final_signal = [], []
        for idx, col in enumerate(x_coords):
            column = img[:, col]
            nz_idx = np.where(column > 0)[0]
            if nz_idx.size == 0:
                continue

            if second_deriv[idx] > 0.5:
                sel_idx = nz_idx[:5]
            elif second_deriv[idx] < -0.5:
                sel_idx = nz_idx[-5:]
            else:
                sel_idx = nz_idx

            weights  = column[sel_idx].astype(float)
            centroid = np.average(sel_idx, weights=weights)
            x_out.append(col)
            final_signal.append(centroid)

        return np.array(x_out), np.array(final_signal)

    def fill_gaps(self, x_coords, y_coords, method='linear'):
        """Interpolate missing columns to produce a dense, evenly-spaced signal.

        Parameters
        ----------
        x_coords : np.ndarray of int
            Sparse column indices from :meth:`binarize_signal`.
        y_coords : np.ndarray of float
            Corresponding centroid Y-values.
        method : str, optional
            Interpolation kind passed to ``scipy.interpolate.interp1d``
            (default ``'linear'``).

        Returns
        -------
        x_full : np.ndarray of int
            Dense array ``[x_coords[0], …, x_coords[-1]]``.
        y_interp : np.ndarray of float
            Interpolated Y-values at every integer column.
        """
        x_full = np.arange(x_coords[0], x_coords[-1] + 1)
        interpolator = interp1d(x_coords, y_coords, kind=method, fill_value="extrapolate")
        y_interp = interpolator(x_full)
        return x_full, y_interp

    def smooth_signal(self, sig, window_length=7, polyorder=4):
        """Apply a Savitzky-Golay low-pass filter to a 1-D signal.

        Parameters
        ----------
        sig : np.ndarray
            Input signal array.
        window_length : int, optional
            Filter window size in samples (must be odd, default ``7``).
        polyorder : int, optional
            Polynomial order (default ``4``).

        Returns
        -------
        np.ndarray
            Smoothed signal of the same length as *sig*.
        """
        return savgol_filter(sig, window_length, polyorder)

    def extract_signals(self):
        """Convert mask grid cells into calibrated voltage time-series.

        For each cell in ``self.grid``:

        1. Calls :meth:`binarize_signal` to get column-centroid Y-values.
        2. Trims 5 edge samples, fills gaps, and smooths with Savitzky-Golay.
        3. Converts pixel Y-coordinates to millivolts using
           ``self.baseline`` and ``self.volt_per_pixel``.
        4. Converts column indices to seconds using ``self.time_per_pixel``.
        5. Resamples uniformly at 500 Hz via linear interpolation.

        Sets ``self.signal_grid`` with the same structure as ``self.grid``
        but each cell additionally contains:

        * ``'time'``   – np.ndarray of time values in seconds.
        * ``'signal'`` – np.ndarray of voltage values in millivolts.
        """
        signal_grid = []
        sample_rate = 500

        for row_idx, row in enumerate(self.grid):
            baseline_y = self.baseline[row_idx]
            row_signals = []

            for cell in row:
                x_coords, sig = self.binarize_signal(cell['signal'])

                x_coords = x_coords[5:-5]
                sig = sig[5:-5]

                x_coords, sig = self.fill_gaps(x_coords, sig, method='linear')
                sig = self.smooth_signal(sig)

                signal_volts = (baseline_y - sig) * self.volt_per_pixel
                x_seconds = x_coords * self.time_per_pixel

                duration = x_seconds[-1] - x_seconds[0]
                num_samples = round(duration * sample_rate) + 1
                resampled_time = np.linspace(x_seconds[0], x_seconds[-1], num_samples)

                interpolator = interp1d(x_seconds, signal_volts, kind='linear',
                                        fill_value="extrapolate")
                resampled_signal = interpolator(resampled_time)

                row_signals.append({
                    'lead': cell['lead'],
                    'time': resampled_time,
                    'signal': resampled_signal
                })

            signal_grid.append(row_signals)

        self.signal_grid = signal_grid

    # ------------------------------------------------------------------
    # Metrics & comparison
    # ------------------------------------------------------------------

    def sliding_metrics(self, signal_a, signal_b, return_aligned_signals=False):
        """Compute alignment-aware similarity metrics between two signals.

        Slides the shorter signal across the longer one and finds the offset
        that maximises the Pearson correlation.  Reports RMSE and SNR at
        the best-aligned position.

        Parameters
        ----------
        signal_a, signal_b : array-like
            The two signals to compare (need not be the same length).
        return_aligned_signals : bool, optional
            If ``True``, also return the aligned signal pair (default ``False``).

        Returns
        -------
        max_corr : float
            Maximum Pearson correlation found across all offsets.
        rmse : float
            Root mean squared error at the best offset.
        snr : float
            Signal-to-noise ratio in dB at the best offset
            (``inf`` if noise power is zero).
        short_signal : np.ndarray
            The shorter signal (only when *return_aligned_signals* is ``True``).
        best_window : np.ndarray
            The aligned window from the longer signal
            (only when *return_aligned_signals* is ``True``).
        """
        len_a, len_b = len(signal_a), len(signal_b)
        if len_a > len_b:
            long_signal, short_signal = signal_a, signal_b
        else:
            long_signal, short_signal = signal_b, signal_a

        len_long, len_short          = len(long_signal), len(short_signal)
        max_corr, best_corr_offset   = -1, 0

        for i in range(len_long - len_short + 1):
            window = long_signal[i:i + len_short]
            corr, _ = pearsonr(window, short_signal)
            if corr > max_corr:
                max_corr         = corr
                best_corr_offset = i

        best_window  = long_signal[best_corr_offset:best_corr_offset + len_short]
        rmse         = mean_squared_error(best_window, short_signal)
        signal_power = np.mean(np.square(short_signal))
        noise_power  = np.mean(np.square(best_window - short_signal))
        snr          = (10 * np.log10(signal_power / noise_power)
                        if noise_power > 0 else np.inf)

        if return_aligned_signals:
            return max_corr, rmse, snr, short_signal, best_window
        return max_corr, rmse, snr

    def calculate_metrics_ptb(self, plot_signals=True, per_lead_scores=None):
        """Evaluate digitization quality against a WFDB ground-truth record.

        For each lead in ``self.signal_grid``, aligns the extracted signal
        to the corresponding WFDB signal using :meth:`sliding_metrics` and
        records Pearson r, RMSE, SNR, and p-value.  Requires ``self.wfdb_path``
        to point to a valid WFDB record.

        Sets the following instance attributes after completion:

        * ``self.average_pearson`` – mean Pearson r across all leads.
        * ``self.average_rmse``    – mean RMSE (mV).
        * ``self.average_snr``     – mean SNR (dB).
        * ``self.average_pval``    – mean p-value.

        Parameters
        ----------
        plot_signals : bool, optional
            Unused placeholder (default ``True``).
        per_lead_scores : dict or None, optional
            If provided, per-lead metrics are appended here for leads where
            Pearson r > 0.60.  Expected structure:
            ``{lead_name: {'pearson': [], 'rmse': [], 'snr': [], 'pval': []}}``.
        """
        record = wfdb.rdrecord(self.wfdb_path)
        avg_pearson, avg_rmse, avg_snr, avg_pval = [], [], [], []

        for row in self.signal_grid:
            for cell in row:
                if 'lead' not in cell or 'signal' not in cell:
                    continue
                try:
                    lead_index = record.sig_name.index(cell['lead'])
                except ValueError:
                    continue

                wfdb_signal    = record.p_signal[:, lead_index]
                wfdb_signal    = wfdb_signal[~np.isnan(wfdb_signal)]
                voltage_signal = np.array(cell['signal'])

                pearson_val, rmse, snr, sig1, sig2 = self.sliding_metrics(
                    voltage_signal, wfdb_signal, return_aligned_signals=True
                )

                try:
                    pearson, pval = pearsonr(sig1, sig2)
                except Exception:
                    pearson, pval = np.nan, np.nan

                cell.update({'pearson': pearson, 'rmse': rmse,
                             'snr': snr,         'pval': pval})

                avg_pearson.append(pearson)
                avg_rmse.append(rmse)
                avg_snr.append(snr)
                avg_pval.append(pval)

                if per_lead_scores is not None and pearson > 0.60:
                    lead = cell['lead']
                    if lead not in per_lead_scores:
                        per_lead_scores[lead] = {
                            'pearson': [], 'rmse': [], 'snr': [], 'pval': []
                        }
                    per_lead_scores[lead]['pearson'].append(pearson)
                    per_lead_scores[lead]['rmse'].append(rmse)
                    per_lead_scores[lead]['snr'].append(snr)
                    per_lead_scores[lead]['pval'].append(pval)

        self.average_pearson = np.mean(avg_pearson)
        self.average_rmse    = np.mean(avg_rmse)
        self.average_snr     = np.mean(avg_snr)
        self.average_pval    = np.mean(avg_pval)

    def plot_signals(self, title='', plot_wfdb=False):
        """Plot extracted signals, optionally overlaid with the WFDB ground truth.

        Generates one figure per lead.  When *plot_wfdb* is ``True``, the
        extracted signal and the best-aligned WFDB window are plotted
        together for visual comparison.

        Parameters
        ----------
        title : str, optional
            Figure title applied to every subplot (default ``''``).
        plot_wfdb : bool, optional
            If ``True``, overlay the ground-truth WFDB signal.  Requires
            ``self.wfdb_path`` and a loaded ``record`` in scope
            (default ``False``).
        """
        for row in self.signal_grid:
            for cell in row:
                voltage_signal = cell['signal']
                plt.figure(figsize=(10, 4))
                if not plot_wfdb:
                    plt.plot(voltage_signal, linewidth=1.5)
                else:
                    lead_index  = record.sig_name.index(cell['lead'])
                    wfdb_signal = record.p_signal[:, lead_index]
                    wfdb_signal = [x for x in wfdb_signal if not np.isnan(x)]
                    _, _, _, sig1, sig2 = self.sliding_metrics(
                        voltage_signal, wfdb_signal, return_aligned_signals=True
                    )
                    plt.plot(sig1, label='Extracted Signal', linewidth=1.5)
                    plt.plot(sig2, label='Ground Truth',     linewidth=1.5)
                plt.title(title)
                plt.legend()
                plt.xlabel("Time (ms)")
                plt.ylabel("Voltage (mV)")
                plt.tight_layout()
                plt.show()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_signals_as_wfdb(self, record_name, directory='.'):
        """Export extracted signals as a WFDB record (.hea + .dat files).

        Pads all lead signals to the same length with NaN (converted to 0),
        then writes a 500 Hz, 16-bit WFDB record in millivolts.

        Parameters
        ----------
        record_name : str
            Base name for the output files (no extension).
        directory : str, optional
            Destination directory (default ``'.'``).
        """
        signals, lead_names, max_length = [], [], 0
        for row in self.signal_grid:
            for cell in row:
                if 'signal' in cell and 'lead' in cell:
                    sig = np.array(cell['signal'])
                    signals.append(sig)
                    lead_names.append(cell['lead'])
                    max_length = max(max_length, len(sig))

        padded = [np.pad(s, (0, max_length - len(s)),
                         mode='constant', constant_values=np.nan)
                  for s in signals]
        signal_array = np.nan_to_num(np.array(padded).T, nan=0.0)

        wfdb.wrsamp(
            record_name=record_name,
            fs=500,
            units=['mV'] * len(lead_names),
            sig_name=lead_names,
            p_signal=signal_array,
            fmt=['16'] * len(lead_names),
            write_dir=directory,
        )

    def save_signals_as_csv(self, record_name, directory='.'):
        """
        Save ECG signals to CSV. Each column = one lead, each row = one time sample.
        """
        signals, lead_names, max_length = [], [], 0
        for row in self.signal_grid:
            for cell in row:
                if 'signal' in cell and 'lead' in cell:
                    sig = np.asarray(cell['signal'], dtype=float)
                    signals.append(sig)
                    lead_names.append(cell['lead'])
                    max_length = max(max_length, len(sig))

        if not signals:
            raise ValueError("No valid signals found in signal_grid.")

        padded = [np.pad(s, (0, max_length - len(s)),
                         mode='constant', constant_values=np.nan)
                  for s in signals]
        df = pd.DataFrame(np.vstack(padded).T, columns=lead_names)

        os.makedirs(directory, exist_ok=True)
        csv_path = os.path.join(directory, f"{record_name}.csv")
        df.to_csv(csv_path, index=False, float_format="%.6f")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self):
        """Run all digitization stages end-to-end for a single ECG image.

        Executes the following stages in order, with per-stage wall-clock
        timing printed to stdout via :meth:`_timed`:

        1. **Image loading & preprocessing** — ``load_image`` + ``preprocess_image``.
        2. **Segmentation** — ``segment_leads`` → ``make_segmentation_mask``
           → ``find_row_centers`` → ``get_roi``.
        3. **Sequential YOLO detections** — ``extract_lead_boxes``,
           ``extract_lead_name_boxes``, ``extract_reference_pulses``.
           (Sequential, not parallel, to avoid a PyTorch thread-pool deadlock.)
        4. **Retry loop** — if any of the three detections is empty, the
           image is reloaded at the next fallback target size from
           ``[2000, 2100, 1900, 2200, 1800, 1700]`` px and stages 1–3 are
           re-run.  Raises ``RuntimeError`` if all sizes fail.
        5. **Calibration** — ``get_reference_scale``.
        6. **Grid & signal extraction** — ``make_grid`` → ``extract_signals``.

        Raises
        ------
        RuntimeError
            If all fallback target sizes yield at least one empty detection.
        """
        FALLBACK_SIZES = [2000, 2100, 1900, 2200, 1800, 1700]

        for target_size in FALLBACK_SIZES:

            # -- Image loading & preprocessing --
            self._timed("load_image",       self.load_image, self.image_path,
                        target_size=target_size)
            self._timed("preprocess",       self.preprocess_image)

            # -- Segmentation (sequential: each step depends on the previous) --
            self._timed("segment_leads",    self.segment_leads)
            self._timed("make_mask",        self.make_segmentation_mask)
            self._timed("find_row_centers", self.find_row_centers)
            self._timed("get_roi",          self.get_roi)

            # -- Three YOLO calls sequentially --
            self._timed("extract_lead_boxes",       self.extract_lead_boxes)
            self._timed("extract_lead_name_boxes",  self.extract_lead_name_boxes)
            self._timed("extract_reference_pulses", self.extract_reference_pulses)

            # -- Check all three detections; retry at next size if any is empty --
            missing = []
            if len(self.reference_pulses) == 0:
                missing.append("pulses")
            if len(self.lead_bboxes) == 0:
                missing.append("lead boxes")
            if len(self.lead_name_bboxes) == 0:
                missing.append("lead names")

            if not missing:
                if target_size != FALLBACK_SIZES[0]:
                    print(f"    ↳ All detections found at target_size={target_size}",
                          flush=True)
                break
            else:
                print(f"    ↳ Missing {', '.join(missing)} at target_size={target_size},"
                      f" retrying...", flush=True)

        else:
            raise RuntimeError(
                f"Detections incomplete at all target sizes {FALLBACK_SIZES}. "
                f"Last missing: {', '.join(missing)}."
            )

        # -- Remainder of pipeline (sequential) --
        self._timed("get_reference_scale", self.get_reference_scale)
        self._timed("make_grid",           self.make_grid)
        self._timed("extract_signals",     self.extract_signals)
